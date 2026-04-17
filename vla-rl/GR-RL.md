# GR-RL：面向长时序高精度机器人操作

> 论文：**GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation**  
> 这份笔记重点整理 **训练部分**，尤其是：
> 1. 离线 critic 如何训练成一个 task progress evaluator  
> 2. 这个 progress evaluator 如何用于数据过滤  
> 3. 在线 RL 为什么不直接在动作空间探索，而是在噪声/潜变量空间优化  
> 4. 论文中出现的关键公式分别在做什么

---

## 0. 核心结论

GR-RL 不是“直接拿 demonstration 做行为克隆”，也不是“从零开始在线强化学习”。  
它的训练链条更准确地写成：

$$\text{原始示范/轨迹} \rightarrow \text{offline RL critic} \rightarrow \text{progress critic} \rightarrow \text{数据过滤} \rightarrow \text{高质量数据集} \rightarrow \text{BC / flow matching} \rightarrow \text{离线策略} \rightarrow \text{online RL} \rightarrow \text{部署对齐后的最终策略}$$

其中训练最关键的两步是：

- **离线阶段**：用 sparse reward + TD3+BC 训练一个 critic，把它当成任务进度估计器；
- **在线阶段**：不在动作空间直接加噪声，而是在 flow policy 的噪声空间中训练一个 **noise predictor** 做结构化探索。

---

# 1. 方法要解决的问题

论文的出发点很明确：对于穿鞋带这种 **长时序 + 高精度 + 柔性物体 + 双臂协作** 的任务，单纯 imitation learning 往往不够。

原因主要有两个。

## 1.1 demonstration 并不“干净”

即使是熟练 teleoperator 采集的数据，也会包含：

- 犹豫动作
- 试探动作
- 失败后修正动作
- 多次错误尝试

如果直接模仿全量示范数据，policy 会学到大量 **suboptimal fragments**，这会显著拉低高精度任务的上限。

## 1.2 训练行为和部署行为不一致

论文认为，对于高精度控制，部署时实际执行的动作与离线训练看到的动作并不完全一致。例如：

- 动作 chunk 在执行前后可能经过平滑或重规划
- flow / diffusion 类动作生成模型的采样过程与监督学习分布之间有偏差
- 长时序误差会逐步累积

因此，必须做 **offline-to-online adaptation**，而不能只停留在离线 BC。

---

# 2. 整体训练流程

GR-RL 的训练可以拆成三段：

## Stage A：用 offline RL 训练任务进度模型，并做数据过滤

- 训练一个 critic $Q_\phi(o_t, l, s_t, a_t)$
- 把 critic 的输出均值当成 progress $\rho_t$
- 找出 progress 会在短时间内显著下降的样本，判定为 suboptimal
- 从行为克隆数据集中删掉这些样本

## Stage B：在过滤后的数据上做离线策略学习

- 用 behavior cloning / flow matching 训练 VLA policy
- 再结合论文中的镜像增强（morphological symmetry augmentation）提升泛化

## Stage C：做在线 RL 微调

- 在共享 VLM backbone 后添加一个 **noise predictor** $\pi_{\theta'}$
- 它不直接输出动作，而是输出 flow policy 的初始噪声 $\epsilon_t$
- 通过优化噪声空间中的策略，使生成出来的动作更符合在线成功轨迹

---

# 3. 数据过滤：为什么 critic 可以被当成任务进度估计器

这一部分对应论文的 **3.1 Data Filtering with a Learned Task Progress Evaluator**。

## 3.1 critic 的训练目标

论文明确写到：他们使用 **TD3+BC** 来训练 critic，并定义一个稀疏奖励：

$$r(o_t, l, s_t, a_t)=\begin{cases}\gamma^{T-t}\,\mathbb{I}(\tau), & t > T-k \\ 0, & t \le T-k\end{cases}\tag{1}$$

### 公式解释

这里：

- $o_t$：视觉观测
- $l$：语言指令
- $s_t$：机器人状态
- $a_t$：当前动作
- $\tau$：整条轨迹
- $\mathbb{I}(\tau)$：轨迹是否成功的指示函数
- $T$：轨迹总长度
- $\gamma$：discount factor
- $k$：只在轨迹末尾一段时间内给奖励的窗口长度

### 这个奖励的含义

这个 reward 设计不是在每一步都给 dense progress label，而是：

- 只有在轨迹后段才给奖励；
- 如果整条轨迹最终成功，那么末尾若干步会得到折扣后的正奖励；
- 否则全程为 0。

换句话说，**它把“最终是否完成任务”这个信号，通过 temporal-difference 学习往前传播**。

### 为什么这样做

因为论文认为，显式人工标注“哪一步是好动作”反而会引入主观偏置。  
所以他们不做手工 progress 标注，而是让 offline RL 自己从成功/失败轨迹中学出一个 progress estimator。

## 3.2 为什么 critic 的值可以看成 progress

论文的关键判断是：

> 在这种稀疏终局奖励设定下，critic 学到的 $Q$-value 可以被解释为“从当前 transition 往后最终成功的前景”，也就是 task progress。

训练完 critic 后，作者对每个 transition 计算：

$$\rho_t := \mathrm{mean}\!\left(Q_\phi(o_t, l, s_t, a_t)\right)\tag{2}$$

这里 $\rho_t$ 被称为 progress。

### 公式解释

因为文中使用的是 **categorical / distributional critic**，所以 $Q_\phi$ 不是一个单点标量，而是一个分布。  
他们取这个分布的均值，作为当前 transition 的 progress score：

- $\rho_t$ 大：当前动作更有利于走向成功
- $\rho_t$ 小：当前动作离成功更远，或后续更可能失败

### 它为什么不是普通 value，而是“任务进度”

直观上，如果一条轨迹中某一步做错了，后续成功概率会明显下降，那么该步附近的 $\rho_t$ 会突然下降。  
作者在图中展示的就是这种现象：teleoperator 犯错时，progress 会突然掉一截。

## 3.3 如何构造更多失败轨迹

文中还有一个很关键但容易被忽略的训练细节：**失败数据增强**。

因为大部分收集到的 demonstration 最终是成功的，所以如果只拿成功轨迹训练 critic，正负样本不平衡，critic 会很难学到“哪里出错了”。

论文的做法是：

- 在每条成功 demonstration 中，手工标出若干个 **retry keyframes**
- 假设这些关键帧为 $m_i,\;0\le i<M$
- 对于一条成功轨迹 $\tau_{0:T}$，构造前缀失败轨迹 $\tau_{0:m_i}$

于是，一条成功轨迹除了原始成功样本之外，还能额外产生 $M$ 条 hindsight failure trajectories。

### 这么做的作用

这一步非常重要，因为它让 critic 看到：

- 哪些前缀看起来“很像在做任务”
- 但实际上并没有完成任务

也就是让 critic 学会区分：

- 真正朝成功推进的片段
- 看似合理但最终需要 retry 的片段

这是 progress model 能成立的关键数据基础。

## 3.4 数据过滤规则到底是什么

有了 progress 序列 $\rho_t$ 后，论文定义：

> 如果样本 $(o_t, l, s_t, a_t)$ 在接下来的一个短窗口 $\rho_{t:t+k}$ 中出现了超过阈值 $\delta$ 的 value drop，则将该样本判为 suboptimal，并从 BC 数据集中剔除。

虽然截图里没有把这个规则写成一个独立公式，但文字定义可以写成：

$$\exists j \in \{t,\dots,t+k\},\quad \rho_t - \rho_j > \delta \;\Longrightarrow\; (o_t,l,s_t,a_t)\ \text{is suboptimal}$$

### 这个规则的含义

它不是看“当前 progress 绝对值高不高”，而是看：

> 从当前开始，短时间内 progress 会不会明显下跌。

这比直接按 $\rho_t$ 大小阈值筛选更合理，因为：

- 某些任务阶段本来 progress 就低，但动作是正确的；
- 真正有害的动作，往往体现为 **短时内造成 progress 崩塌**。

### 过滤完之后怎么训练 policy

论文这里很直接：  
过滤完后，policy $\pi_\theta$ 只需要在高质量数据集上做 **behavior cloning** 即可。

也就是说，这一段 RL 的作用主要不是直接更新 actor，而是：

> **先用 RL 学一个“会打分”的 critic，再用这个 critic 清洗 imitation 数据。**

---

# 4. 在线 RL：为什么不直接在动作空间探索

这一部分对应你给的第二张图，也是 GR-RL 训练中最容易被概括得过于粗糙的地方。

论文明确说：

> 对于毫米级精度任务，直接在 wrist pose 或 joint position 上加噪声，几乎不可能成功。

所以他们不采用常规 action-space noise，而是做：

$$\text{structured exploration in latent/noise space}$$

也就是在 **flow policy 的噪声空间** 中探索。

## 4.1 noise predictor 的作用

论文在共享 VLM backbone 后添加一个 **noise predictor** $\pi_{\theta'}$，让它预测 action DiT 的初始噪声 $\epsilon_t$。

这意味着：

- 原来的离线 flow policy $\pi_\theta$ 仍负责“如何从噪声逐步生成动作”
- 新加的 $\pi_{\theta'}$ 负责“给这个生成过程一个更有利的起点噪声”

所以在线 RL 优化的核心对象不是动作 $a_t$，而是噪声 $\epsilon_t$。

### 为什么这样更合理

因为 flow / diffusion policy 本身已经把“合法动作 manifold”学出来了。  
如果直接在动作空间乱加噪声，会很容易跑出这个 manifold，导致动作失真。  
而在噪声空间调节，相当于：

> 在“模型已经学会的动作生成流形”内部做更结构化、更平滑的探索。

## 4.2 噪声策略的优化目标

论文给出的 noise predictor 损失为：

$$\mathcal{L}(\pi_{\theta'})=\mathbb{E}_{(o_t,l,s_t)\sim \mathcal{D}}\left[-\,Q_{\phi'}(o_t,l,s_t,\epsilon_t)+c\,\max\left(\frac{1}{2}\|\epsilon_t\|^2-\beta,\;0\right)\right],\qquad \epsilon_t\sim \pi_{\theta'}(o_t,l,s_t)\tag{3}$$

### 公式逐项解释

这里：

- $\pi_{\theta'}$：噪声空间策略，也就是 noise predictor
- $\epsilon_t$：它输出的初始噪声
- $Q_{\phi'}(o_t,l,s_t,\epsilon_t)$：定义在噪声空间中的 critic
- $c$：正则项权重
- $\beta$：阈值，用于限制噪声不要偏离标准高斯太远

这个 loss 的两部分含义很清楚：

#### 第一项：$-Q_{\phi'}$

最小化这个项，相当于最大化噪声空间 critic 给出的价值。  
也就是说，noise predictor 会学着输出那些能导向更高回报动作的噪声。

#### 第二项：噪声范数惩罚

$$c\,\max\left(\frac{1}{2}\|\epsilon_t\|^2-\beta,\;0\right)$$

这一项是一个 hinge-style regularization。它的作用是：

- 如果噪声范数没超过阈值 $\beta$，不惩罚；
- 一旦偏离“原始标准正态分布”太远，就开始惩罚。

### 为什么需要这个正则项

论文明确说，这是为了避免 noise predictor 输出过于极端的噪声，从而让 flow policy 生成 **脱离离线训练分布的任意动作**。

所以这一步不是无约束地“让在线 RL 自己找任何高回报 noise”，而是：

> 在接近原始 normal prior 的局部区域内，学习一个更有利的噪声偏移。

## 4.3 为什么还要蒸馏一个噪声空间 critic

论文进一步写到：为了避免在 policy optimization 时通过 flow model 反向传播，他们额外蒸馏了一个定义在噪声空间的 critic：

```math
Q_{\phi'}(o_t,l,s_t,\epsilon_t)
```

它的训练目标是：

```math
\mathcal{L}(Q_{\phi'})=\mathrm{cross\_entropy}\Big(Q_{\phi'}(o_t,l,s_t,\epsilon_t),\; Q_\phi\big(o_t,l,s_t,\pi_\theta(o_t,l,s_t\mid \epsilon_t)\big)\Big),\quad \epsilon_t \sim \begin{cases}\mathcal{N}(0,1), & \text{with prob. }0.5 \\ \pi_{\theta'}(o_t,l,s_t), & \text{otherwise}\end{cases}\tag{4}
```

### 公式解释

这里：

- $Q_\phi(o_t,l,s_t,a_t)$：原始动作空间中的 critic
- $\pi_\theta(o_t,l,s_t\mid \epsilon_t)$：给定噪声 $\epsilon_t$ 后，flow policy 生成的动作
- $Q_{\phi'}(o_t,l,s_t,\epsilon_t)$：试图直接预测“这份噪声最终会带来多好的动作”的噪声空间 critic

#### 它在做什么

它本质上是在学习一个映射：


```math
\epsilon_t \mapsto Q_\phi(\text{由 }\epsilon_t\text{ 生成的动作})
```

这样优化 noise predictor 时，就不需要每次都穿过完整 flow model 再回传梯度，而是可以直接利用 $Q_{\phi'}$ 提供训练信号。

### 为什么输入噪声有 50% 从标准高斯采样

论文特别强调，与参考实现不同，他们为了保证噪声空间覆盖更好，蒸馏 $Q_{\phi'}$ 时：

- 以 0.5 概率从原始 $\mathcal{N}(0,1)$ 采样
- 以 0.5 概率从当前 noise predictor $\pi_{\theta'}$ 采样

这一步的作用是平衡：

- **coverage**：避免 $Q_{\phi'}$ 只在当前策略附近有效
- **on-policy relevance**：又要让它对当前 noise predictor 常访问的区域足够准确

---

# 5. 在线训练的数据缓冲区设计

这一点也很工程化，但很关键。

论文在线训练时维护两个 buffer：

- **off-policy buffer**
- **on-policy buffer**

并且从两者中 **均匀采样 batch**。

## 5.1 为什么要两个 buffer

这是为了兼顾：

- 稳定性：保留历史数据，不让训练完全追着最新噪声跑
- 适应性：又要确保模型能快速适应最近策略产生的数据分布

## 5.2 warm start 怎么做

在线训练开始前，作者先用离线训练好的 checkpoint 跑若干 online rollouts，把 off-policy buffer warm up。  
这和 Warm-start RL 的思路一致：避免在线阶段一开始就因为样本太少而崩掉。

## 5.3 为什么不把 teleoperated trajectories 混进在线 buffer

论文明确说，他们**有意不把 teleoperated trajectories 混入在线 buffer**，因为这会引入 dynamics mismatch。

意思是：

- online RL 希望学的是“policy 自己在当前控制闭环下执行出来的行为”
- teleoperation 数据对应的是另一套控制/交互分布
- 混在一起会让 online actor/critic 的目标不干净

## 5.4 on-policy buffer 只保留最近两个 checkpoint 的数据

论文还做了一个很有针对性的设计：

- on-policy buffer 只存最近两个 checkpoint 生成的轨迹
- 更旧的数据会被推入 off-policy buffer

这样做的目的，是让 on-policy 区域更贴近当前策略，而不是被陈旧 rollout 淹没。

---

# 6. GR-RL 的训练逻辑可以怎么理解

从训练视角看，GR-RL 可以理解成下面这个分层结构。

## 6.1 第一层：critic 不是直接拿来提 actor，而是先拿来清洗数据

这和很多 actor-critic 方法不一样。  
在离线阶段，critic 的首要作用不是直接做 policy improvement，而是：

> 先判断 demonstration 里哪些片段值得学，哪些片段应该删掉。

所以它更像一个 **task-progress-driven data curation module**。

## 6.2 第二层：actor 的离线训练仍然是 imitation / flow matching

也就是说，过滤完数据后，policy 本身还是在做 supervised learning。  
这一阶段的收益来自：

- 数据质量提升
- augmentation 提升
- VLA backbone 的条件建模能力

## 6.3 第三层：真正的 RL actor 更新发生在噪声空间

在线阶段，GR-RL 才开始显式优化一个策略，但这个策略不是原始动作策略，而是：

$$\pi_{\theta'}(\epsilon_t \mid o_t,l,s_t)$$

即 **噪声策略**。

最终动作仍由原始 flow policy 生成，只不过现在它的初始噪声被在线 RL 学会了“往更好的区域偏”。

---

# 7. 对比一句话总结

如果用一句更准确的话概括 GR-RL 的训练：

> **离线阶段用 TD3+BC 学到一个 distributional progress critic，用它过滤 demonstration 中会导致 progress 崩塌的片段；在线阶段不直接优化动作，而是在 flow policy 的噪声空间中训练一个 noise predictor，并用噪声空间 critic 做高效策略优化。**

---

# 8. 更算法式的流程

## 输入

- demonstration 轨迹（多数成功，但带有 suboptimal fragments）
- 语言指令 $l$
- 观测 $o_t$
- 机器人状态 $s_t$
- 动作 $a_t$

## 离线阶段

1. 用式 (1) 的 sparse reward 训练 TD3+BC critic $Q_\phi$
2. 计算每个 transition 的 progress：
   $$\rho_t=\mathrm{mean}\!\left(Q_\phi(o_t,l,s_t,a_t)\right)$$
3. 若未来短窗口中出现超过阈值 $\delta$ 的 progress drop，则删除该 transition
4. 用过滤后的数据集训练 flow/BC policy $\pi_\theta$

## 在线阶段

5. 固定或继承离线学到的 flow policy 主干
6. 添加 noise predictor $\pi_{\theta'}$，输出初始噪声 $\epsilon_t$
7. 用式 (4) 蒸馏噪声空间 critic $Q_{\phi'}$
8. 用式 (3) 优化 noise predictor
9. 在 on-policy / off-policy 混合 buffer 上持续迭代

## 输出

- 一个适用于部署闭环、适合高精度长时序操作的最终策略

---

# 9. 你最该抓住的三点

## 9.1 progress 不是人工标的，是 offline RL 学出来的

它本质上是 sparse reward 下的 value / Q 结构化结果。

## 9.2 数据过滤不是“只保留成功轨迹”

而是保留那些不会在短期内触发明显 progress drop 的局部 transition。

## 9.3 在线 RL 优化的是噪声，不是直接优化动作

这是 GR-RL 在高精度 manipulation 中最有技术含量、也最不同于普通 RL 的地方。
