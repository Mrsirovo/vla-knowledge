# MEM：VLA 的多尺度具身记忆（长短期记忆）

> 论文：**MEM: Multi-Scale Embodied Memory for Vision Language Action Models**  
> Physical Intelligence 等，2026-03 · arXiv:2603.03596  
> 落地形态：π0.6-MEM；后续 [`../vla-rl/pi0.7.md`](../vla-rl/pi0.7.md) 继续沿用该视频历史编码  
>
> 这份文档重点整理：
>
> 1. 为什么“把全部历史帧塞进 context”不可行  
> 2. **短期视频记忆** 与 **长期语言记忆** 如何分工  
> 3. 高效 video encoder / 高层策略如何更新 \(m_t\)  
> 4. 带来的能力：长程任务、遮挡、in-context 纠错

---

# 1. 一句话概括

MEM 的核心主张是：

> **机器人记忆需要多粒度、多模态：几秒内的动力学与遮挡要用稠密视觉历史；十几分钟的任务进度要用高度压缩的自然语言摘要。二者一起，才能在实时延迟约束下做长达约 15 分钟的任务。**

不是再训一个 RNN latent memory，而是：

| 尺度 | 模态 | 典型用途 |
|------|------|----------|
| Short-term | 稠密图像序列（经视频编码器压缩） | 自遮挡、再抓取、近期失败后的策略微调 |
| Long-term | 自然语言 memory \(m_t\) | 菜谱做到哪一步、哪面柜台擦过、门是否还开着 |

---

# 2. 问题：长程控制卡在记忆，而不是单个技能

当技能本身已经较强时，瓶颈变成：

- 任务叙事：清理厨房 / 备菜进行到哪一步  
- 部分可观测：物体被手臂挡住、抽屉已关上看不见内容  
- 避免重复犯错：刚刚抓滑了，下一拍要换握姿  

朴素做法——把过去所有观测都喂给 Transformer——在“数十分钟 × 多相机 × 高频”下会爆：

- token 与算力爆炸  
- 实时控制延迟超预算（文中对照约 300ms 量级实时门槛）  
- 单一模态很难同时保留“毫米级空间细节”和“分钟级语义进度”

因此需要 **按时间尺度选表示**，而不是 one-size-fits-all。

---

# 3. 整体分解：高层记语义，低层看短视频

MEM 把策略因子化为：

```math
\pi(a_{t:t+H},\,\ell_{t+1},\,m_{t+1}
\mid o_{t-T:t},\,m_t,\,g)
\;\approx\;
\pi_{\mathrm{LL}}(a_{t:t+H}\mid o_{t-K:t},\,\ell_{t+1},\,g)
\;
\pi_{\mathrm{HL}}(\ell_{t+1},\,m_{t+1}\mid o_t,\,m_t,\,g)
```

含义：

- \(g\)：总任务语言目标  
- \(\ell_{t+1}\)：当前子任务指令（与 π0.5 / π0.7 的 subtask 一脉相承）  
- \(m_t\)：**长期语言记忆**（语义事件摘要）  
- \(o_{t-K:t}\)：短窗口稠密观测（\(K \ll T\)），经 **video encoder** 压进低层策略  

关键新点不只是“有高层策略”，而是：

> **\(\pi_{\mathrm{HL}}\) 同时决定下一步子任务 \(\ell\)，以及如何把 \(m_t\) 更新成 \(m_{t+1}\)**——主动选择“记什么、怎么压缩”。

---

# 4. 长期记忆：语言摘要 \(m_t\)

## 4.1 形式

\(m_t\) 是自然语言段落，例如：

```text
m_t:   I placed a plate in the cabinet and moved to the counter.
m_{t+1}: I placed a plate in the cabinet, moved to the counter, and picked up a bowl.
```

高层策略根据当前观测与旧记忆写出新记忆；不再需要的细节会被压缩或删掉（例如“三个不同颜色碗”→“三个碗放进右上柜”）。

## 4.2 训练标签怎么来

对带 subtask 标注 \(\ell_{0:T}\) 的 episode：

1. 把已执行子任务序列 + 成功/失败指示交给外部 LLM  
2. 要求：只保留对未来执行仍相关的最小信息  
3. 得到 \(m_t \rightarrow m_{t+1}\) 监督，训练 \(\pi_{\mathrm{HL}}\)

压缩的动机：

- 推理更快（context 短）  
- **减轻 train–test shift**：示范里子任务常只出现一次；推理时可能反复失败同一子任务。若把所有历史子任务指令裸拼接，会看到训练从未见过的“重复失败串”；摘要记忆则在成功前可不更新，分布更稳

消融也支持：naive 拼接历史指令 ≪ 可压缩的预测式 language memory。

---

# 5. 短期记忆：时空可分的视频编码器

## 5.1 为什么不能逐帧塞进 VLM backbone

图像编码往往是 VLA 推理的大头。帧数一增，延迟迅速越过实时门槛。  
目标：给策略 **数秒～数十秒** 的稠密视觉历史，但送进 backbone 的 token 数仍接近单帧 VLA。

## 5.2 架构要点

在标准 ViT / SigLIP 类视觉编码器上扩展，而不引入一整套新参数：

1. 各帧先 patchify  
2. 多数层：帧内 **空间双向 attention**（与普通 ViT 相同）  
3. 每隔若干层（文中约每 4 层）：对同一 patch 位置跨时间做 **因果时间 attention**  
4. 复杂度从朴素时空联合 \(O(n^2 K^2)\) 降到可分的 \(O(K n^2 + n K^2)\)  
5. **上层丢掉过去帧的 patch token**，只把“当前时刻已融合历史信息”的表征交给 VLA backbone  

因此：

- 输出 token 数 ≈ 无记忆单帧 VLA  
- \(K=1\) 时与原 VLM 视觉塔初始化对齐（时间位置编码在 \(t=0\) 为零）  
- 可用预训练 ViT 权重启动，再学“把历史写进当前帧表征”

本体感觉历史若仍用文本离散化会爆 token；π0.6-MEM 改为对每帧状态做 **线性投影**，\(K\) 个状态 → \(K\) 个连续 token。

## 5.3 训练时窗口 vs 部署时拉长

- 预训练示例：约 **6** 帧（当前 + 5 过去），stride 约 **1s**  
- 后训练可拉长到更多帧（文中实验可到约 **18** 帧 / **54s** 量级视觉记忆）  
- 与 LLM 长上下文类似：短窗预训练，长窗后适应

---

# 6. 落到 π0.6-MEM（及与 π0.7 的关系）

π0.6-MEM 实例：

- Backbone：Gemma3-4B 级 VLM + SigLIP 视觉  
- Action expert：~860M flow matching；Knowledge Insulation（expert 梯度不回传 backbone）  
- 离散 FAST token + 连续 flow 双路径监督  
- 多相机（最多约 4 路）、448²  
- 数据混合：teleop、rollout、人类纠错、VLM 任务、视频-语言任务等  

[`pi0.7`](../vla-rl/pi0.7.md) 明确建立在 **π0.6 + MEM 视频历史** 之上，并再叠加 metadata / 子目标图像等 conditioning；因此理解 MEM，是读懂 π0.6→π0.7 记忆侧演进的前置。

---

# 7. 能力与消融（读实验时抓什么）

## 7.1 真正需要“记十几分钟”的任务

- **Recipe setup**：按长提示取齐厨具食材，记住已取什么、关没关门  
- **Clean kitchen**：擦哪面台、碗洗没洗皂、前后是否都冲过  
- 另有烤芝士三明治等长程烹饪流程  

无记忆的强通才（如裸 π0.6）在这些设定上明显吃力；**视频短期 + 语言长期** 都去掉会掉点，说明两路不可互相替代。

## 7.2 In-context adaptation（短任务也受益）

即使地平线不长，短期视觉记忆也能：

- 手臂挡住目标时仍知道物体在哪  
- 抓取失败后立刻改策略（换角度 / 重试），而不必等整段 episode 级重训  

这是“记忆 ≠ 只服务超长任务”的重要结论。

## 7.3 与其它记忆路线的对比直觉

| 路线 | 长处 | 短板 |
|------|------|------|
| 全历史稠密帧 | 信息全 | 算力 / 延迟不可扩展 |
| 只语言 / 只本体 / 只点轨迹 | 压缩好 | 丢掉精细空间与动力学 |
| 只 keyframe | 能拉长 | 过稀则估不好动态 |
| **MEM** | 短视觉密 + 长语言稀 | 依赖摘要质量与双塔配合 |

关于 causal confusion（记忆导致抄历史动作）：作者认为大规模多样数据下，MEM 可在不加专门辅助目标时仍表现良好；若数据更窄，仍可考虑与防抄袭辅助目标组合。

---

# 8. 实践理解：记忆的最小心智模型

```text
当前要做什么？
  ← 高层：g + m_t + 当前观测 → 新子任务 ℓ + 更新后的 m_{t+1}

当前怎么做精细动作？
  ← 低层：短窗视频记忆编码 + ℓ → action chunk
```

若只记三点：

1. **问题**：长程任务需要多尺度记忆；全帧历史不实时  
2. **方法**：短期 = 时空可分视频塔压进当前表征；长期 = 可压缩语言摘要由高层策略维护  
3. **收益**：约 15 分钟级任务叙事 + 遮挡鲁棒 + 失败后的 in-context 适应；并成为 π0.6-MEM / π0.7 的记忆底座

---

# 9. 参考文献与资源

- Torne, Pertsch, et al., *MEM: Multi-Scale Embodied Memory for Vision Language Action Models*, arXiv:2603.03596, 2026.  
- [博客：VLAs with Long and Short-Term Memory](https://www.pi.website/research/memory)  
- [论文 PDF](https://www.pi.website/download/Mem.pdf)  
- 相关：[`../vla-rl/pi0.7.md`](../vla-rl/pi0.7.md)、[`grounding_and_planning.md`](./grounding_and_planning.md)、[`efficiency_acceleration.md`](./efficiency_acceleration.md)
