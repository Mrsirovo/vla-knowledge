# RL Token (RLT)：用紧凑接口给 VLA 做快速在线 RL

> 论文 / 博客：**RL Token: Bootstrapping Online RL with Vision-Language-Action Models**  
> Physical Intelligence，2026-03  
> 相关前置：[`pi0.6*.md`](./pi0.6*.md)（Recap，整模型离线/批量 RL）、[`GR-RL.md`](./GR-RL.md)（噪声空间在线 RL）  
>
> 这份文档重点整理：
>
> 1. 为什么“整模 RL”和“从小模型从零 RL”都不适合精巧接触任务的小时级改进  
> 2. **RL token** 是什么、怎么从 VLA 里压出来  
> 3. 小 actor-critic 如何在 token 上做 chunked、锚定 VLA 的在线更新  
> 4. 为何聚焦 critical phase，以及和 Recap / π0.7 的分工

---

# 1. 一句话概括

RLT 的核心目标是：

> **冻结大 VLA，只抽出一个紧凑的 RL token 作为状态接口；在其上训练轻量 actor-critic，用几十分钟到几小时的真实机器人数据，把任务里最难、最需要毫米级精度的阶段做得更快更稳。**

它不是用 RL 重训整个 foundation model，而是：

```text
VLA 提供感知先验 + 参考动作
  → RL token 压缩内部表征
  → 小网络做局部动作精修
```

---

# 2. 问题：精巧操作卡住在“最后一毫米”

通才 VLA 往往能完成任务的前半段（抓起螺丝刀、拿到扎带），但在：

- 螺丝对齐与拧入
- 扎带穿入锁扣
- 网线 / 充电器插入

这类 **接触丰富、公差极小** 的阶段会变慢、重试、失败。  
示范数据本身也常常在这些阶段噪声大、策略不一致，纯 BC 上限不够。

自然想法是上 RL，但现实约束很紧：

| 路线 | 问题 |
|------|------|
| 像 Recap 一样更新整个 VLA | 算力与样本都重，不适合“几小时内针对性修好一个阶段” |
| 像 SERL 一样从小型视觉编码器从零 RL | 样本效率可以，但丢掉了 VLA 的语义与行为先验 |
| 残差 / 噪声空间方法 | 有用，但常受单步动作、调参、或探索范围限制 |

因此问题变成：

> **如何既借用 VLA 的表征与行为，又保持小型 on-robot RL 的速度与样本效率？**

---

# 3. 核心思想：给 VLA 开一个 RL 接口

RLT 把系统拆成两层：

1. **冻结的 VLA**：看图、懂指令、给出参考 action chunk  
2. **可在线更新的小 actor-critic**：只看到压缩后的 RL token（+ 本体感觉），在参考动作附近做编辑

关键装置是 **RL token** \(z_{rl}\)：

- 从 VLA 最终层 token embeddings 读出
- 经小型 encoder-decoder bottleneck 压成固定小向量（文中约 **1×2048**）
- 训练目标是：用 \(z_{rl}\) 重建原始 VLA embeddings，逼它保留任务相关信息

之后在线 RL **不再回传大模型**，只训小头。

---

# 4. 阶段一：适配 VLA，露出 RL token

## 4.1 构造方式

设预训练 VLA 对观测 \(s\)、语言 \(\ell\) 产出最终层 embeddings：

```math
z = f(s,\ell;\theta_{\mathrm{vla}}) = \{z_1,\ldots,z_M\}
```

追加一个可学习 special embedding \(e_{rl}\)，送入轻量 encoder \(g_\phi\)：

```math
z_{rl}
=
g_\phi\big([z_{1:M},\,e_{rl}]\big)_{M+1}
```

再用 decoder \(d_\phi\) + 线性头 \(h_\phi\)，在 stop-gradient 的 \(\bar z_i=\mathrm{sg}(z_i)\) 上做自回归重建：

```math
\mathcal{L}_{\mathrm{ro}}
=
\mathbb{E}_{\mathcal{D}}
\left[
\sum_{i=1}^{M}
\big\|
h_\phi\big(d_\phi([z_{rl},\bar z_{1:i-1}])\big)_i
-
\bar z_i
\big\|_2^2
\right]
```

可选地，同时用一小份任务 demo 对 VLA 做 SFT（权重 \(\alpha\)）：

```math
\min_{\phi,\theta_{\mathrm{vla}}}
\;
\mathcal{L}_{\mathrm{ro}}(\phi)
+
\alpha\,\mathcal{L}_{\mathrm{vla}}(\theta_{\mathrm{vla}})
```

训完后 **\(\theta_{\mathrm{vla}}\) 与 \(\phi\) 全部冻结**，在线阶段只使用 \(z_{rl}\)。

## 4.2 为什么要 bottleneck，而不是随便抽一层

Transformer 内部表征维度高、层多，直接拿来做 RL 状态：

- 小 critic 难拟合
- 更新贵
- 不清楚哪一层对接触控制最有用

重建式 bottleneck 的作用是：

> **强迫单一向量保留“下游还能还原 VLA 理解”的信息，同时小到能被轻量 actor-critic 高效消费。**

---

# 5. 阶段二：在 RL token 上做 chunked 在线 actor-critic

## 5.1 状态与动作接口

RL 输入状态：

```math
x = (z_{rl},\, s^{p})
```

其中 \(s^{p}\) 是关节角 / 末端位姿等本体感觉（任务相关）。

动作不是单步 50Hz 控制量，而是 **action chunk** \(a_{1:C}\)，且通常 \(C < H\)（\(H\) 为 VLA 原 chunk，如 50）。  
这样：

- 与 VLA 的时间抽象对齐
- 缩短稀疏成功奖励下的有效决策地平线
- 比逐步残差更容易做时间 credit assignment

## 5.2 Critic：chunk 级 TD

稀疏奖励：episode 结束由人标成功 / 失败，\(r_T\in\{0,1\}\)。

```math
\mathcal{L}_Q
=
\mathbb{E}_{(x,a_{1:C},x')\sim\mathcal{B}}
\big[
\big(\hat Q - Q_\psi(x,a_{1:C})\big)^2
\big]
```

```math
\hat Q
=
\sum_{t'=1}^{C}\gamma^{t'-1} r_{t'}
+
\gamma^{C}
\mathbb{E}_{a'\sim\pi_\theta}
\big[
Q_{\psi'}(x',a')
\big]
```

实现上接近 TD3 风格（target network、延迟/双 Q 等工程细节见原文）。

## 5.3 Actor：条件在 VLA 参考 chunk 上，并锚定它

Actor 输出高斯 chunk，并显式吃 VLA 采样出的参考动作 \(\tilde a_{1:C}\sim\pi_{\mathrm{vla}}\)：

```math
\pi_\theta(a_{1:C}\mid x,\tilde a_{1:C})
=
\mathcal{N}\big(\mu_\theta(x,\tilde a_{1:C}),\,\sigma^2 I\big)
```

目标：

```math
\mathcal{L}_\pi(\theta)
=
\mathbb{E}
\big[
-Q_\psi(x,a_{1:C})
+
\beta\,\|a_{1:C}-\tilde a_{1:C}\|_2^2
\big]
```

含义：

- 第一项：提高 critic 认为更好的 chunk  
- 第二项：别离开 VLA 太远（正则化 RL / BC anchor）  
- 条件在 \(\tilde a\) 上：把在线 RL 变成 **局部编辑**，而不是在高维 chunk 空间从零搜索  
- 顺带保留 VLA 多峰动作分布里被采样到的那个 mode（单峰高斯自己很难覆盖）

## 5.4 Reference-action dropout

若总是喂 \(\tilde a\) 又总正则向 \(\tilde a\)，早期 critic 还不准时，actor 会退化成“复制 VLA”。

做法：batch 里随机把参考 chunk 置零，迫使 actor 维持独立生成通路；等 critic 有用后，再学会在值得偏离时偏离。

---

# 6. 完整系统循环

概念流程：

```text
1) 用任务 demo 训练 RL token（可选 SFT VLA）
2) Warmup：用冻结 VLA rollout 填 replay
3) Online：
     VLA 出参考 chunk + RL token
     actor 出执行 chunk（可被人接管）
     稀疏成功/失败标注
     高 UTD 的 off-policy actor-critic 更新
4) 可选：再短训 VLA，让它学会何时把控制交给 RL head
```

实践要点：

- Replay 混合：VLA warmup、RL rollout、人类干预  
- 异步收集与学习；较高 update-to-data（文中约 5）  
- Chunk 内可 stride 子采样，增加 transition 数量  
- **Targeted critical phase**：整段任务仍由 base VLA 跑容易部分；人选择何时切到 RL 精修最难阶段，把样本与 TD 信用集中在瓶颈上

这与“整任务端到端 Recap”互补：RLT 刻意做 **小时级、阶段级** 精修。

---

# 7. 实验里应抓住的结论

任务（均需毫米～亚毫米级对齐）：

- 电动螺丝刀拧 M3 螺丝  
- 双臂穿扎带  
- 网线插入  
- 充电器插入

主要结果形态：

- 相对 base VLA，关键阶段吞吐与成功率显著提升；最难阶段速度最高约 **3×**  
- 有的任务（如 Ethernet）最终策略中位时长可 **快过人类 teleop**  
- Ethernet 一类任务可在约 **15 分钟有效机器人数据**（总 wall-clock 更长，含 reset）内看到明显吞吐爬升  
- 相对 HIL-SERL / 单步残差 / 纯噪声空间 RL / 去掉 token 或 chunk 或 BC regularizer 等消融，完整 RLT 更稳、更强

消融直觉：

| 去掉什么 | 典型后果 |
|----------|----------|
| RL token → 普通 ResNet | 丢掉 VLA 内部任务表征，精巧阶段变差 |
| Chunk → 单步 | 高频控制下信用分配更难，且难与 VLA 接口对齐 |
| BC regularizer (\(\beta=0\)) | 探索易漂离可靠先验 |
| Pass-through（不喂 \(\tilde a\)） | 失去“编辑 VLA 提案”的结构，学习更难 |

---

# 8. 和 Recap、GR-RL、π0.7 怎么放一起看

```text
Recap / π*0.6
  更新大 VLA；advantage 条件提取；长程大规模改进

RL Token (本文)
  冻结大 VLA；压缩表征 + 小头在线 RL；精修 critical phase

GR-RL
  也偏“别直接在动作空间瞎探索”
  但主路径是 progress filtering + 在 flow 噪声空间学 predictor

π0.7
  用 metadata/subgoal 条件把含 RL 经验的多样数据蒸馏进通才模型
  部署后若仍有“最后一毫米”瓶颈，仍可用 RLT 局部在线补一刀
```

更短的对照：

| | Recap | RLT |
|--|-------|-----|
| 更新对象 | 整模 VLA | 小 actor-critic |
| 数据规模 | 大、可多轮迭代 | 分钟～小时 |
| 典型目标 | 长程吞吐 / 鲁棒 | 接触阶段精度与速度 |
| 与 VLA 关系 | 把 RL 信号写进同一生成模型 | VLA 当先验与表征源 |

---

# 9. 实践理解：它算哪种 RL

算标准的 **off-policy actor-critic**，但工程上做了三件 VLA 特化：

1. **表征**：RL token bottleneck，而不是 ResNet-from-scratch  
2. **动作**：chunk 级决策，对齐 VLA  
3. **先验**：参考动作 conditioning + \(\beta\) 锚定 + dropout 防抄袭

因此更准确的一句话是：

> **用 VLA 引导的、面向关键阶段的轻量在线 RL，而不是 foundation model 的全参数 RL fine-tune。**

---

# 10. 伪代码（便于对照公式）

```python
# Stage A: expose RL token
for batch in demos:
    z = freeze(vla).final_embeddings(obs, lang)
    z_rl = encoder([z, e_rl])
    loss = reconstruct_embeddings(decoder(z_rl), stopgrad(z))
    loss += alpha * vla_sft_loss  # optional
    optim_phi.step(loss)
freeze(vla, encoder)

# Stage B: online RL
replay = warmup_rollouts(vla)
while robot_training:
    z_rl = encoder(vla.final_embeddings(obs, lang))
    a_ref = vla.sample_action_chunk(obs, lang)[:C]
    a = human_chunk if intervened else actor(z_rl, proprio, a_ref)
    execute(a); store(replay)
    for _ in range(G):  # high UTD
        update_critic_TD(replay)          # Eq. L_Q
        update_actor_Q_minus_bc(replay) # Eq. L_pi, with ref dropout
```

---

# 11. 参考文献与资源

- Xu et al., *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*, 2026.  
- [博客：Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt)  
- [论文 PDF](https://www.pi.website/download/rlt.pdf) / [arXiv](https://arxiv.org/abs/2604.23073)  
- 相关：[`pi0.6*.md`](./pi0.6*.md)、[`pi0.7.md`](./pi0.7.md)、[`GR-RL.md`](./GR-RL.md)
