# GRAPE：通过偏好对齐提升机器人策略泛化

> 论文：**GRAPE: Generalizing Robot Policy via Preference Alignment**  
> 这份文档按“问题 → 核心思想 → 方法 → 公式 → 训练流程 → 实验结论”的顺序整理。  
> 重点放在 **训练机制**，尤其是：
>
> 1. 为什么作者不用纯 SFT，也不直接上显式 reward RL  
> 2. TPO（Trajectory-wise Preference Optimization）到底在优化什么  
> 3. GCPG（Guided-Cost Preference Generation）如何自动合成偏好数据  
> 4. multi-stage cost 为什么能支持任务完成、安全、效率等不同对齐目标  
>

---

# 1. 论文要解决什么问题

这篇论文关注的是：**如何让 Vision-Language-Action (VLA) 模型不仅会模仿专家示范，还能更好地泛化到未见任务，并按用户指定目标进行对齐**。

作者指出，现有 VLA 方法普遍有两个结构性问题。

## 1.1 只学成功示范，泛化性差

主流 VLA 微调通常基于 supervised fine-tuning（SFT），即只在成功 demonstration 上做行为克隆。  
这样学到的是：

- “在训练分布里的成功动作长什么样”
- 而不是“什么样的轨迹整体更优、失败模式是什么、如何在新场景中做取舍”

因此，一旦测试场景发生变化，比如：

- 换物体
- 换背景
- 换操作方式
- 换语言表达
- 换更复杂的目标约束

纯 SFT 的 VLA 很容易掉性能。

## 1.2 demonstration 中隐含了多种目标，但 SFT 不会分辨

真实示范往往同时包含多种隐含偏好：

- 任务完成
- 路径更短
- 避免碰撞
- 保持安全距离
- 动作更平稳

但 SFT 只会把这些都当作“应该模仿的行为”，并不会显式区分：

> 这条轨迹为什么更好？  
> 是因为成功率更高，还是更安全，还是更高效？

所以如果用户想把策略对齐到某个新目标，例如“更安全”或“更省步数”，SFT 没有自然的接口来完成这件事。

---

# 2. GRAPE 的核心思想

GRAPE 的核心思想可以概括成一句话：

> **不用人工逐条写 reward，也不只做成功轨迹模仿，而是先自动构造轨迹级偏好，再用偏好优化的方式微调 VLA。**

这套方法由两部分组成：

## 2.1 TPO：Trajectory-wise Preference Optimization

把整条轨迹当作优化对象，而不是只看单步动作。  
核心直觉是：

- 长时序机器人操作的好坏，很多时候必须在**整条轨迹层面**才能判断；
- 只看 step-level action 容易被局部噪声误导；
- 成功与失败的区别，往往也是轨迹级别的模式差异。

所以 GRAPE 用的是 **trajectory-level preference alignment**。

## 2.2 GCPG：Guided-Cost Preference Generation

但偏好优化需要“哪条轨迹更好”的 supervision。  
人工给大量机器人轨迹成对排序，成本太高。于是 GRAPE 进一步提出：

- 先把复杂任务拆成多个时间阶段
- 再为每个阶段找关键空间 keypoints
- 再让一个强大的语言模型自动生成 cost function
- 最终把整条轨迹打成一个综合分数，再从中选出 chosen / rejected pair

所以它的完整逻辑是：

```math
\text{采样轨迹}
\rightarrow
\text{自动打分}
\rightarrow
\text{构造偏好对}
\rightarrow
\text{做轨迹级偏好优化}
```

---

# 3. 基础形式：SFT 与 RL 视角

论文先把标准 VLA 的 SFT 写成下面的形式。

```math
L_{\mathrm{SFT}}
=
-
\sum_{(\zeta, q)\in D}
\sum_{t=1}^{T}
\log p(a_t \mid o_t, q; \pi_\theta)
```

这里：

- `q` 是任务指令
- `o_t` 是第 `t` 步观测
- `a_t` 是动作
- `\zeta = \{o_1, a_1, \dots, o_T, a_T \mid q\}` 是整条轨迹
- `\pi_\theta` 是 VLA policy

这个目标的本质是：**最大化专家动作的似然**。  
问题在于它只会复制数据分布，而不会学“轨迹级偏好”。

所以作者转向一个带 KL 正则的 RL 目标。

```math
\max_{\pi_\theta}
\mathbb{E}_{\zeta \sim \pi_\theta}[r_\phi(\zeta)]
-
\beta D_{\mathrm{KL}}[\pi_\theta(\zeta)\,\|\,\pi_{\mathrm{ref}}(\zeta)]
```

这里：

- `r_\phi(\zeta)` 是定义在整条轨迹上的 reward
- `\pi_{\mathrm{ref}}` 是参考策略，通常就是 SFT 初始化得到的 base policy
- `\beta` 控制新策略偏离参考策略的程度

这个目标想表达的是：

> 在尽量不偏离原始 SFT policy 太远的前提下，提高更优轨迹的概率。

---

# 4. TPO：Trajectory-wise Preference Optimization

这一部分是论文的核心优化目标。

---

## 4.1 从 RL 目标到轨迹级 reward 重参数化

论文沿用了 DPO/偏好优化一类方法的经典推导，把轨迹 reward 重写成策略比值形式：

```math
r(\zeta, q)
=
\beta
\log \frac{\pi_\theta(\zeta \mid q)}{\pi_{\mathrm{ref}}(\zeta \mid q)}
+
\beta \log Z(\zeta)
```

其中：

- `Z(\zeta)` 是与归一化相关的项
- 真正关键的是 `\log \frac{\pi_\theta}{\pi_{\mathrm{ref}}}` 这一项

这个式子的意义是：

> 如果某条轨迹在当前策略下比在参考策略下更可能出现，那么它应当对应更高 reward。

因此，reward 不需要显式由另一个 reward network 来学，也可以通过“相对参考策略的偏好提升”隐式建模出来。

---

## 4.2 Bradley-Terry 偏好模型

接着，论文对同一起始状态下的两条轨迹 `\zeta_w` 和 `\zeta_l` 建立偏好概率：

```math
P(\zeta_w \succ \zeta_l)
=
\frac{\exp(r(\zeta_w, q))}
{\exp(r(\zeta_w, q)) + \exp(r(\zeta_l, q))}
```

这里：

- `\zeta_w` 是 preferred / chosen trajectory
- `\zeta_l` 是 less preferred / rejected trajectory

这就是标准的 Bradley-Terry 偏好模型。它的直觉是：

> 两条轨迹谁更好，不是由绝对分数直接决定，而是由它们 reward 的相对大小决定。

---

## 4.3 TPO 损失函数

把上面的 reward 重参数化代入 Bradley-Terry 模型后，论文得到 TPO 的训练目标：

```math
L_{\mathrm{TPO}}
=
-
\mathbb{E}_{(\zeta_w,\zeta_l)\sim D}
\left[
\log \sigma
\left(
\beta
\left(
\log \frac{\pi_\theta(\zeta_w)}{\pi_{\mathrm{ref}}(\zeta_w)}
-
\log \frac{\pi_\theta(\zeta_l)}{\pi_{\mathrm{ref}}(\zeta_l)}
\right)
\right)
\right]
```

这里 `\sigma` 是 sigmoid。

### 这个损失在做什么

这个损失的作用非常明确：

- 如果当前策略相比参考策略，更倾向于 chosen trajectory 而不是 rejected trajectory，那么 loss 会下降；
- 如果当前策略没有把 chosen 轨迹和 rejected 轨迹区分开，loss 就会变大。

所以 TPO 的本质是：

> **提升好轨迹的相对概率，压低差轨迹的相对概率。**

---

## 4.4 为什么它是 trajectory-wise

论文进一步把轨迹概率写成逐步动作概率的乘积：

```math
\pi(\zeta, q)=\prod_{t=1}^{T}\pi(a_t \mid o_t, q)
```

因此有：

```math
\log \frac{\pi_\theta(\zeta, q)}{\pi_{\mathrm{ref}}(\zeta, q)}
=
\sum_{t=1}^{T}
\log
\frac{\pi_\theta(a_t \mid o_t, q)}
{\pi_{\mathrm{ref}}(a_t \mid o_t, q)}
```

这意味着：

- 优化对象是**整条轨迹**
- 但梯度仍然可以自然回传到每一步的 state-action pair 上

这也是 TPO 相比“直接做轨迹黑盒排序”的好处：  
它虽然在轨迹层面对齐偏好，但训练仍然能落到 step-wise action likelihood 上。

---

# 5. 为什么 GRAPE 不直接做人类偏好标注

因为人工给机器人轨迹成对排序太贵，而且很难扩展到：

- 任务完成
- 安全
- 效率
- 韧性
- 不同任务类型

所以论文进一步提出 GCPG：**Guided-Cost Preference Generation**，自动构造偏好。

---

# 6. GCPG：Guided-Cost Preference Generation

GCPG 的作用是：

> 自动给每条轨迹打一个“符合目标程度”的分数，再按分数排序生成偏好对。

这一部分可以拆成三步。

---

## 6.1 多阶段时间分解

复杂 manipulation task 往往不是一个原子动作，而是多个阶段组合而成。  
例如：

- 抓取物体
- 移动到目标区域
- 放置物体

因此论文先用一个 stage decomposer `MD` 把轨迹拆成多个连续阶段：

```math
\{\zeta_1, \dots, \zeta_S\} = MD(\zeta, q)
```

其中每个阶段可以写成：

```math
\zeta_i = \{(o_t^i, a_t^i)\}_{t=1}^{T_i}
```

这里：

- `S` 是阶段数
- `\zeta_i` 是第 `i` 个阶段的子轨迹
- `T_i` 是该阶段长度

### 为什么要做阶段分解

因为“整条轨迹好不好”太难直接写 cost。  
但如果先拆成子阶段，就更容易定义：

- 抓得好不好
- 搬运过程是否安全
- 放置是否到位

所以 GRAPE 不是直接对整条轨迹一把梭打分，而是：

> **先分阶段，再分阶段计算 cost，最后聚合。**

---

## 6.2 每阶段的 keypoints 与 cost function

在每个阶段，GRAPE 再使用视觉语言模型提取关键 keypoints，然后提示一个强大的 LLM 自动生成 cost function。

记第 `i` 个阶段的关键点集合为 `\{\kappa^{S_i}\}`，相应阶段代价记为：

```math
C_{S_i}(\{\kappa^{S_i}\})
```

论文强调，**cost 越低，说明越符合当前对齐目标**。

例如，对于任务完成目标，cost 可能与：

- 末端执行器到目标物体中心的距离
- 物体到目标容器的距离

有关。

对于安全目标，cost 则可能与：

- 机械臂到障碍物的距离
- 是否发生接触或碰撞

有关。

对于效率目标，cost 则可能与：

- 路径长度
- 冗余动作
- 轨迹步数

有关。

也就是说，GRAPE 的对齐目标并不是写死的，而是通过 cost function 可配置化。

---

## 6.3 外部 reward：阶段 cost 的聚合

论文没有把所有阶段 cost 简单线性求和，而是设计了一个乘法形式的外部 reward：

```math
R_{\mathrm{ext}}(\zeta)
=
\prod_{i=1}^{S}
e^{-C_{S_i}(\{\kappa^{S_i}\})}
```

这个式子的含义是：

- 单个阶段 cost 小，则该阶段贡献接近 1
- 单个阶段 cost 大，则该阶段会把整体 reward 明显压低

### 为什么用指数和乘法

作者的直觉是：时序阶段之间是有因果依赖的。  
如果早期阶段已经做得很差，那么后续阶段就不应被过于乐观地看待。乘法聚合能更自然地表达这种“前面坏，后面整体就很难好”的关系。

所以这个 `R_{\mathrm{ext}}` 可以理解为：

> 基于阶段 cost 的、面向目标约束的外部对齐分数。

---

# 7. GCPG 的完整 reward 组合

为了让偏好生成更稳定，GRAPE 不只依赖 external reward。  
论文进一步引入了两个额外量：

- policy 的 self-reward
- 最终 success indicator

组合成最终的 GCPG reward：

```math
R_{\mathrm{GCPG}}(\zeta)
=
\lambda_1 R_{\mathrm{self}}(\zeta)
+
\lambda_2 R_{\mathrm{ext}}(\zeta)
+
\lambda_3 I_{\mathrm{success}}(\zeta)
```

这里：

- `R_{\mathrm{self}}`：模型对这条轨迹本身的“自我评分”
- `R_{\mathrm{ext}}`：由多阶段 cost 聚合得到的外部分数
- `I_{\mathrm{success}}`：任务是否成功的二值指标
- `\lambda_1,\lambda_2,\lambda_3`：三项权重

---

## 7.1 Self-reward

论文定义：

```math
R_{\mathrm{self}}(\zeta)
=
\log \pi(\zeta, q)
=
\log \left(\prod_{t=1}^{T}\pi(a_t \mid o_t, q)\right)
```

也就是轨迹在当前策略下的对数似然。

### 这项在做什么

它的作用是：

- 给模型一个“自己认为更自然/更可实现”的偏好信号
- 避免 external judge 完全主导，导致偏好过于脱离当前 policy 分布

可以理解为一种 self-consistency regularization。

---

## 7.2 Success indicator

论文定义：

```math
I_{\mathrm{success}}(\zeta)
=
\begin{cases}
1, & \text{if } \zeta \text{ is successful} \\
0, & \text{otherwise}
\end{cases}
```

这项很直接，就是最终任务是否完成。

### 为什么还需要它

因为仅靠 cost 函数，有时会出现：

- 轨迹局部看起来很合理
- 但实际上没有完成任务

所以 success indicator 负责把**任务最终完成**这个全局目标拉回来。

---

## 7.3 为什么要三项一起用

GCPG 里三项各自解决不同问题：

- `R_self`：防止偏好过于偏离当前策略可行域
- `R_ext`：把用户指定目标（完成、安全、效率）显式编码进去
- `I_success`：保证最终任务完成性不被忽略

所以 `R_GCPG` 不是一个随意相加的 heuristic，而是：

> **策略自身可行性 + 外部目标约束 + 最终任务完成性** 的综合分数。

---

# 8. 从 reward 到偏好对

有了 `R_GCPG(\zeta)` 之后，GRAPE 的偏好生成流程就很直接了：

1. 对每个任务在线采样很多条轨迹；
2. 计算每条轨迹的 `R_GCPG(\zeta)`；
3. 按分数排序；
4. 取 top-m 与 bottom-m 组成 chosen / rejected pairs；
5. 用这些 pair 做 TPO 微调。

也就是说，GRAPE 不需要人工逐条说“这条轨迹比那条更好”，而是通过一套自动评分机制，自己构造 preference data。

---

# 9. 迭代式偏好优化

论文不是离线一次性构造 preference data，而是采用 **iterative preference optimization**。

算法流程可以概括为：

## Step 1：从 base VLA 采样轨迹
当前策略 `\pi_\theta` 在多个任务上在线采样，得到当前轮轨迹集 `D_k`。

## Step 2：对每条轨迹做阶段分解和成本计算
对每条轨迹：

- 分解阶段
- 计算每阶段 cost
- 得到 `R_ext`
- 计算 `R_self`
- 判断 `I_success`
- 聚合成 `R_GCPG`

## Step 3：按 `R_GCPG` 排序并构造偏好对
对同一任务下的轨迹排序，取 top-m 和 bottom-m 配对。

## Step 4：用 TPO 更新策略
把这些 pair 输入 TPO loss，得到更新后的策略。

## Step 5：继续下一轮
用新策略再次在线采样，重复这一过程。

这使得策略能不断在自己生成的数据上被重新对齐，而不是只依赖初始示范数据。

---

# 10. GRAPE 为什么比直接 RL 或 DPO 更适合机器人

## 10.1 比显式 reward RL 更省人工

显式 RL 需要手工 reward，代价高，而且不同对齐目标需要重写 reward。  
GRAPE 通过 stage decomposition + keypoint cost generation，把这个过程半自动化了。

## 10.2 比纯 DPO 更适合长时序轨迹

文本 DPO 通常针对单个输出序列直接比较。  
机器人任务里，作者认为真正重要的是：

- 整条轨迹是否完成任务
- 中间阶段是否符合空间约束
- 是否更安全、更省步数

所以他们做的是 **trajectory-wise preference optimization**，而不是 token-level preference alignment。

## 10.3 比纯 SFT 更能学失败模式

SFT 只看成功 demonstration。  
GRAPE 在偏好构造中同时利用成功与失败轨迹的排序，这能让模型显式学到：

- 哪些轨迹虽然看起来像成功动作，但其实整体更差
- 哪些失败模式应该被压低概率

---

# 11. 实验结果怎么理解

论文在真实环境和模拟环境都做了评测。

根据作者报告，GRAPE 相比 SOTA VLA 基线：

- in-domain manipulation task 的成功率提升 **51.79%**
- unseen manipulation task 的成功率提升 **58.20%**
- 在安全对齐目标下，collision rate 降低 **37.44%**
- 在效率对齐目标下，rollout step-length 降低 **11.15%**

这些结果说明两件事：

## 11.1 轨迹级偏好优化确实提升了泛化

也就是说，GRAPE 学到的不只是“更像训练集动作”，而是更偏向全局更优轨迹的倾向。

## 11.2 cost-based alignment 确实能切换目标

它不是只能优化“任务完成率”，还可以通过改 cost function 去对齐：

- 安全
- 效率
- 其他用户指定目标

这说明 GRAPE 的核心价值不只是性能提升，而是**对齐目标的可配置化**。

---

# 12. 论文的优点

## 12.1 问题定义准确

作者抓住了 VLA 当前一个非常现实的问题：  
SFT 可以复现训练分布，但很难在新任务和新目标下稳定泛化。

## 12.2 方法链条完整

GRAPE 不是只提出一个 loss，而是给出了一整套闭环：

```math
\text{在线采样}
\rightarrow
\text{自动打分}
\rightarrow
\text{偏好构造}
\rightarrow
\text{偏好优化}
\rightarrow
\text{再次在线采样}
```

## 12.3 alignment object 可扩展

通过改不同阶段的 cost function，它可以支持：

- task completion
- safety
- cost-efficiency

而不需要重写整套训练框架。

---

# 13. 论文的局限

## 13.1 仍然依赖外部大模型生成阶段和 cost

虽然比纯人工省事，但 stage decomposition、keypoint proposal、cost generation 仍然依赖强大的 VLM / LLM。

## 13.2 偏好质量受 cost 质量影响

如果自动生成的 cost function 不合理，那么整个 preference ranking 也会偏掉。

## 13.3 轨迹级方法的采样成本不低

因为它需要在线采样大量轨迹，再做排序配对，所以训练代价仍然高于纯离线 SFT。

---

# 14. 最终总结

如果只用一句话概括这篇论文：

> **GRAPE 用自动构造的轨迹级偏好替代人工 reward 设计，通过 trajectory-wise preference optimization 把 VLA 从“只会模仿成功示范”推进到“能按全局轨迹优劣和用户目标进行对齐”的阶段。**

再具体一点：

- **TPO** 负责“怎么根据偏好更新策略”
- **GCPG** 负责“怎么自动生成偏好”
- **multi-stage cost** 负责“怎么把任务完成、安全、效率等目标显式编码进去”

所以这篇论文最值得记住的点是：

> 它把机器人 VLA 的对齐问题，从“写 reward”改写成了“自动构造轨迹偏好再做偏好优化”。

---
