# π0.7：可操控的通才机器人基础模型

> 论文 / 博客：**π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities**  
> Physical Intelligence，2026-04  
> 相关前置：[`pi0.6*.md`](./pi0.6*.md)（Recap / advantage-conditioned post-train）、[`models/π₀.₅.md`](../models/π₀.₅.md)  
>
> 这份文档重点整理：
>
> 1. π0.7 相对 π0.6 / π\*0.6 解决了什么问题  
> 2. **diverse multimodal context conditioning** 为什么能吃下混合质量数据  
> 3. 子目标图像、episode metadata、控制模态分别在做什么  
> 4. 它如何把 RL specialist（Recap）的经验**蒸馏**进一个通才模型  
> 5. 推理时如何用 prompt / CFG 操控策略

---

# 1. 一句话概括

π0.7 的核心主张是：

> **不要只做“任务语言 → 动作”的平均模仿；而是在训练时给每条轨迹附上足够细的多模态上下文（怎么做、做得好不好、子目标长什么样），让模型学会按条件切换策略，从而同时吃下 demo、失败轨迹、autonomous RL rollout，并在推理时被 steer 到高质量行为。**

结果上，一个**未经任务级 fine-tune** 的 π0.7，可以在多个灵巧长程任务上达到甚至超过各自用 Recap 训出来的 π\*0.6 specialist；并开始表现出组合泛化与跨 embodiment 迁移。

---

# 2. 问题：为什么朴素混合多样数据会失败

π 系列此前已经证明：

- 大规模 demo + VLM 预训练 → 通才能力（π0 / π0.5）
- Recap 式 advantage conditioning → 单任务高吞吐（π\*0.6）

但仍有两道门槛：

## 2.1 通才模型往往“会，但不稳、不快”

很多技能在训练里见过，但最好成绩通常仍要靠 **task-specific post-train**。  
也就是说：通才模型更像 early LLM 时代的“需要领域微调”，还不是“开箱即用的通才”。

## 2.2 数据越杂，朴素 BC 越容易 mode average

π0.7 想用的数据远比高质量 demo 更广：

- 低质量 / 失败 demonstration
- 策略评估与 autonomous rollout（含失败）
- Recap / π\*0.6 等 RL 训练过程中产生的经验
- 人类 egocentric 视频、网页多模态数据、开源机器人数据

如果不加区分地混训，模型会把：

- 快 vs 慢
- 优 vs 差
- 不同控制模态 / 不同策略风格

平均成一个模糊行为，表现反而变差。

因此关键问题是：

> **如何让模型从“异质、次优、甚至失败”的数据中学到可组合的技能，而不是学到平均后的差策略？**

---

# 3. 核心思想：用上下文消歧，而不是只靠过滤数据

π0.7 的回答不是“只留高质量数据”，而是：

```text
更细的 prompt / context
  → 消歧“这条轨迹在教什么”
  → 允许混合质量数据进入训练
  → 推理时用高质量条件提取行为
```

上下文 \(\mathcal{C}_t\) 不再只是一句任务描述 \(\ell_t\)，而是多模态组合：

| 组件 | 作用 | 典型内容 |
|------|------|----------|
| Task / Subtask 语言 | 说清做什么、当前子步骤 | `peel vegetables` / `pick up the peeler` |
| Episode metadata | 说清做得怎样、多快 | Speed / Quality / Mistake |
| Control mode | 说清动作空间 | `joint` / `ee` |
| Subgoal images | 用图像说清“下一步世界长什么样” | 多视角近未来画面 |

训练目标仍是条件动作似然：

```math
\max_\theta\;
\mathbb{E}_{\mathcal{D}}
\big[
\log \pi_\theta(\mathbf{a}_{t:t+H}\mid \mathbf{o}_{t-T:t},\mathcal{C}_t)
\big]
```

但关键变化在 \(\mathcal{C}_t\)：它把“策略改进信号”从 **后训练 RL 标签**，前移成了 **可在 pretrain / mid-train 里消费的条件**。

一句话对比：

- **π\*0.6 / Recap**：用 critic 算 advantage，再写成 `Advantage: positive/negative` 做条件提取  
- **π0.7**：用更丰富的 strategy metadata + 子目标图像，把包括 RL rollout 在内的多样数据统一条件化，再在推理时用“高质量 metadata”提取行为

---

# 4. 模型架构：在 π0.6-MEM 上扩展上下文

π0.7 大约 **5B** 参数：

- **VLM backbone**：Gemma3 4B（含约 400M vision encoder）
- **Action expert**：约 860M，flow matching 生成连续动作 chunk
- **Memory**：沿用 [MEM](../techniques/MEM.md) 的视频历史编码（时空压缩，历史帧数变化时 token 数固定）

其他实现要点：

- 输入最多 4 路相机（前视、双腕、可选后视），每路最多 6 帧历史
- 子目标最多 3 路图像（通常不含后视）
- 图像 resize 到 \(448\times 448\)
- 本征状态 \(\mathbf{q}_t\) 用线性投影进 backbone（不再像 π0.6 那样离散成 text token）
- Action expert 固定处理 **50** 个动作 token（约 1s @ 50Hz）
- Knowledge Insulation：backbone 用 FAST token 的离散 CE 监督；action expert 可 attend backbone，但梯度不回传到 backbone
- 训练时模拟 RTC（real-time chunking）延迟，适配推理延迟下的平滑执行

---

# 5. Prompt 各组件详解

## 5.1 Subtask instructions

继承 π0.5 的分层思想：

- 总任务 \(\ell_t\)：如 “clean the kitchen”
- 当前子任务 \(\hat{\ell}_t\)：如 “open the fridge door”

作用有两层：

1. **训练时**：给长程轨迹更细的语义分段监督  
2. **推理 / 示教时**：支持人类逐步 language coaching；之后可用这些 coaching 轨迹微调一个 **high-level policy**，让它自动产出 \(\hat{\ell}_t\)

这对组合泛化很关键：新家电任务往往不是“零样本一句搞定”，而是“可被逐步语言带过去”，再蒸馏成自主高层策略。

## 5.2 Subgoal images

语言子任务仍可能不够具体（“打开冰箱”没说怎么抓把手）。  
子目标图像 \(\mathbf{g}_t\) 提供近未来多视角画面，把目标状态直接可视化。

训练 / 推理分工：

- **训练**：真实未来帧 + 世界模型生成帧混合，减轻 train-test gap
- **推理**：轻量世界模型 \(g_\psi\)（基于 BAGEL 类图像生成/编辑模型）根据当前观测与子任务生成 \(\mathbf{g}_t\)

直观效果：动作学习更接近 **inverse dynamics**（已知当前与目标，推断中间动作），对语言跟随与空间 grounding 更友好。

## 5.3 Episode metadata

这是把“次优 / RL / 失败数据”变成可用监督的关键开关。常见字段：

- **Speed**：episode 长度离散化（如每 500 step 一档）
- **Quality**：1–5 分质量
- **Mistake**：当前片段是否犯错

训练时模型看到的是“带标签的行为分布”；  
推理时则固定为：

- Quality = 5
- Mistake = false
- Speed = 该任务长度分布的偏快分位（文中用约 15th percentile）

于是：

> 低质量轨迹不是噪声，而是“Quality: 2 / Mistake: true”条件下的合法样本；  
> 高质量行为则通过推理时的 metadata 被显式召唤出来。

这也是 π0.7 能蒸馏 π\*0.6 Recap 经验的机制：RL specialist 的高吞吐 rollout 带着对应 metadata 进入通才模型。

## 5.4 Control mode

文本标识 \(c \in \{\texttt{joint},\texttt{ee}\}\)，让同一模型覆盖关节空间与末端空间控制，并在测试时按任务选择。

## 5.5 Dropout：训练成“可缺省条件”的 steerable 模型

各组件随机 dropout，使测试时可用任意子集：

- 子目标图像只出现在约 25% batch（有它时训练更快，但不能依赖它永远存在）
- 有子目标时，子任务语言再以一定概率 dropout
- metadata 整体 / 各字段也有小概率 dropout
- control mode 通常不 dropout

这带来两个好处：

1. 部署灵活：可只用语言，也可上齐 subgoal + metadata  
2. 支持对 metadata 做 **classifier-free guidance (CFG)**，在去噪时把动作推向“更快 / 更高质量”条件

---

# 6. 和强化学习 VLA 的关系：蒸馏，而不是替代 Recap

## 6.1 Recap / π\*0.6 解决什么

Recap（见 [`pi0.6*.md`](./pi0.6*.md)）面向**长程任务的大规模经验改进**：

- distributional value
- advantage → `Advantage: positive/negative`
- 在混合 demo / rollout / intervention 上做条件 BC / flow matching

它很强，但通常是 **per-task specialist**。

## 6.2 π0.7 怎么吸收这些 RL 成果

π0.7 并不在同一个通才模型里重跑完整 Recap；而是：

```text
π*0.6 / Recap 在线或离线产生的经验
  → 标注 speed / quality / mistake 等 metadata
  → 作为 π0.7 训练数据的一部分
  → 推理时用高质量 metadata（+ 可选 CFG）提取
```

实验上，**同一个** π0.7 在 laundry / espresso / box building 等任务上，成功率与吞吐可对齐甚至超过各自的 π\*0.6 specialist。

消融也支持这条链：

- 去掉 eval / autonomous 数据 → 吞吐明显掉  
- 去掉 metadata → 无法正确利用混合质量数据，掉点同样显著

## 6.3 和 RL Token 的分工

| 方法 | 尺度 | 目标 |
|------|------|------|
| Recap / π\*0.6 | 整模型、长程、大规模数据 | 全面提升成功率与吞吐 |
| [RL Token](./RL-Token.md) | 冻结 VLA + 小 actor-critic，分钟～小时级 | 精修接触丰富的“最后一毫米” |
| **π0.7** | 通才 pretrain / mid-train | 用条件化把多样经验（含 RL）蒸馏进一个可操控通用模型 |

三者不是互斥：RL 产出经验 → π0.7 蒸馏；部署后仍可用 RLT 对关键阶段做快速在线精修。

---

# 7. 推理时如何 steer

典型 runtime 流程（概念版）：

1. 固定 control mode + 高质量 metadata  
2. High-level policy 或人类给出当前 \(\hat{\ell}_t\)  
3. 世界模型生成 / 刷新 subgoal images（子任务变化或每隔约 4s）  
4. VLA 以 flow matching（少量 denoising steps）生成 50-step chunk，执行前缀  
5. 可选：对 metadata 做 CFG，加强“快 / 高质量”偏好

```text
C = {task, subtask, subgoal images, metadata, control mode}
a ~ π(a | o_history, C)   # 可选 CFG(metadata)
```

---

# 8. 能力与实验要点（读论文时抓什么）

## 8.1 Out-of-the-box 灵巧任务

无需任务级 post-train，对齐 π\*0.6 / SFT specialist；部分任务吞吐更高。

## 8.2 Instruction following

未见厨房 / 卧室中的开放指令跟随显著强于 π0.5 / π0.6；复杂指代、反数据偏见指令上，子目标图像（GC）帮助更大。

## 8.3 Cross-embodiment

例如双臂 UR5e 零样本叠衣：训练数据几乎没有该 embodiment 的叠衣轨迹，仍能迁移；成功率可接近“在源机器人上很熟、但第一次 teleop 目标机”的人类专家零样本水平。

## 8.4 Compositional generalization

新家电（如空气炸锅烤红薯）：

1. 纯零样本语言：部分成功但不稳  
2. 逐步 language coaching：可完成  
3. 用 coaching 数据微调 high-level policy + 世界模型子目标：可更自主执行

这说明泛化更像 LLM 式“组合已见概念”，而不是单靠某一条高度相似 demo。

---

# 9. 实践理解：π0.7 在 VLA-RL 谱系中的位置

```text
π0 / π0.5     通才 BC + 语义分层
     ↓
π0.6 + Recap  用 advantage 条件做经验后训练 → π*0.6 specialist
     ↓
π0.7          用多模态 strategy conditioning
              把 demo / 失败 / RL rollout 统一进一个 steerable generalist
```

若只记三点：

1. **问题**：多样数据会 mode-average；通才往往还要 per-task RL/SFT  
2. **方法**：语言 + metadata + control mode + subgoal 图像，训练时 dropout，推理时高质量条件 + 可选 CFG  
3. **与 RL 的关系**：不替代 Recap / RLT，而是把 RL 经验蒸馏进可开箱使用的通才模型

---

# 10. 参考文献与资源

- Physical Intelligence, *π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities*, 2026.  
- [博客：A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)  
- [论文 PDF](https://www.pi.website/download/pi07.pdf) / [arXiv HTML](https://arxiv.org/html/2604.15483)  
- 相关：[`pi0.6*.md`](./pi0.6*.md)（Recap）、[`RL-Token.md`](./RL-Token.md)、[`../techniques/MEM.md`](../techniques/MEM.md)（长短期记忆）
