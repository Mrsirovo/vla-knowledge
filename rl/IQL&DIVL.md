# Implicit Q-Learning (IQL) 与 Distributional Implicit Value Learning (DIVL)
> 知识笔记：从离线 RL 基础讲到 IQL，再讲 LWD 论文中的 DIVL 扩展。  
> 主要参考：
>
> - Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*, 2021  
> - Wang et al., *Learning while Deploying: Fleet-Scale RL for Generalist Robot Policies*, 2026
---
## 目录
1. [预备：强化学习最小工具箱](#1-预备强化学习最小工具箱)
2. [离线 RL 的核心困难](#2-离线-rl-的核心困难)
3. [IQL：Implicit Q-Learning](#3-iqlimplicit-q-learning)
4. [DIVL：Distributional Implicit Value Learning](#4-divldistributional-implicit-value-learning)
5. [IQL vs DIVL 对照](#5-iql-vs-divl-对照)
6. [自检问题](#6-自检问题)
7. [符号速查表](#7-符号速查表)
---
## 1. 预备：强化学习最小工具箱
### 1.1 交互循环
智能体与环境交互：
1. 观测状态 `s`
2. 选择动作 `a`
3. 获得奖励 `r`，转移到下一状态 `s'`

目标：最大化长期累积回报，而不是单纯模仿演示动作。
### 1.2 回报与折扣
从时刻 `t` 起的折扣回报：
```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```
| 符号 | 含义 |
|------|------|
| `r_t` | 时刻 `t` 的即时奖励 |
| `\gamma \in (0,1]` | 折扣因子；越接近 1 越重视远期奖励 |
### 1.3 状态价值 `V` 与动作价值 `Q`
在策略 `\pi` 下：
```math
V^\pi(s) = \mathbb{E}\big[G_t \mid s_t = s\big]
```
```math
Q^\pi(s,a) = \mathbb{E}\big[G_t \mid s_t = s,\, a_t = a\big]
```
| 函数 | 含义 |
|------|------|
| `V(s)` | 「这个局面大概有多好」 |
| `Q(s,a)` | 「在这个局面下，选这个动作有多好」 |
关系：
```math
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\big[Q^\pi(s,a)\big]
```
### 1.4 Bellman / TD 目标
一步自举形式：
```math
Q(s,a) \approx r + \gamma V(s')
```
右边称为 **TD 目标**（Temporal-Difference target）：用「真实一步奖励 + 对未来的估计」监督当前 `Q`，不必等整局结束。
### 1.5 机器人设定中的 chunk 记号（可选）
很多机器人策略一次输出一段动作（action chunk）：
```math
{a}_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
```
对应 chunk 奖励：
```math
{r}_t = \sum_{i=0}^{H-1} \gamma^i r_{t+i}
```
样本常写成 `(s_t, {a}_t, {r}_t, s_{t+H})`。下文公式在「单步」与「chunk」下含义相同，只是时间粒度不同。
---
## 2. 离线 RL 的核心困难
### 2.1 在线 vs 离线
| 设定 | 数据来源 | 特点 |
|------|----------|------|
| 在线 RL | 当前策略与环境交互 | 策略变了可立刻采新数据 |
| 离线 RL | 固定数据集 `\mathcal{D}=\{(s,a,r,s')\}` | 不能随意试危险/昂贵动作 |
### 2.2 经典 Q-learning 的 `\max` 问题
经典 backup：
```math
y = r + \gamma \max_{a'} Q(s', a')
```
在线、能充分探索时往往有效；**纯离线**时，`\max_{a'}` 可能选到数据中从未出现的动作。神经网络对未见动作常会**乱估很高** → 过估计 → 训练崩溃。
离线 RL 的关键原则：
> **尽量只在数据支撑（in-support）内做策略改进，避免对 OOD 动作做激进最大化。**
IQL 正是为这一设定设计的。
---
## 3. IQL：Implicit Q-Learning
### 3.1 核心思想（人话）
IQL 想同时做到：
1. 仍然用价值学习（利用成功/失败的结果信号，不只模仿动作）
2. **绝不显式**执行 `\max_{a'} Q(s', a')`
做法：
> 不在整个动作空间里找最大 `Q`，  
> 而是在**数据集中已经出现过的动作**上，  
> 学一个「偏向上等水平」的状态价值 `V(s)`，  
> 再用 `V(s')` 去更新 `Q`。
因此叫 **Implicit（隐式）**：改进通过「把 `V` 拟合到高 expectile」隐式完成，而不是显式 `\arg\max_a Q`。
### 3.2 Expectile：IQL 的数学工具
#### 普通均值
最小化 `\mathbb{E}[(x - v)^2]` 得到均值。
#### 非对称平方损失（expectile 损失）
```math
\rho_{\tau,2}(u) = \lvert \tau - \mathbb{I}(u < 0) \rvert \, u^2
```
其中 `u = x - v`：
| 情况 | `u` | 权重 | 含义 |
|------|-------|------|------|
| 真实值比 `v` 大（`v` 估低了） | `u > 0` | `\tau` | |
| 真实值比 `v` 小（`v` 估高了） | `u < 0` | `1-\tau` | |
当 `\tau > 1/2`（如 `0.7`）：
- 「估低了」惩罚更重 → 最优 `v` 被往上推
- 但仍完全由数据中的 `x` 决定，**不会凭空发明更大的 `x`**
直觉：`\tau`-expectile ≈「数据里偏高水平的平均值」，介于均值与 `\max` 之间，且不离开数据支撑。
> 对比：分位数用 `\lvert u \rvert^1`（`p=1`）；expectile 用 `u^2`（`p=2`）。统一写法见第 4.5 节。
### 3.3 IQL 的两个损失
同时学习：
- `Q_\phi(s,a)`：动作价值网络
- `V_\psi(s)`：状态价值网络（**标量**）
- `Q_{\bar\phi}`：`Q` 的 EMA target 网络（稳定 TD 目标）
#### （1）学 `V`：拟合数据集动作 `Q` 值的高 expectile
```math
\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\Big[\rho_{\tau,2}\big(Q_{\bar\phi}(s,a) - V_\psi(s)\big)\Big]
```
| 符号 | 含义 |
|------|------|
| `(s,a)\sim\mathcal{D}` | 只从离线数据采样（只看真实做过的动作） |
| `Q_{\bar\phi}(s,a)` | 该数据动作当前的价值估计 |
| `V_\psi(s)` | 要学的局面价值 |
| `\tau > 1/2` | 让 `V` 偏向数据中较高的那些 `Q` |
**人话：** 同一状态上可能有好动作、坏动作；`V(s)` 不是简单平均，而是「接近好动作那一档」，但仍在数据动作范围内。
#### （2）学 `Q`：用 `V(s')` 做 TD 目标
```math
y = r + \gamma V_\psi(s')
```
```math
\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\big[\big(Q_\phi(s,a) - y\big)^2\big]
```
**人话：** 「做完 `a` 到达 `s'` 后，未来价值用偏乐观但仍 in-support 的 `V(s')` 估计。」注意：**没有** `\max_{a'} Q(s',a')`。
#### Chunk 版（与机器人论文一致）
```math
\mathcal{L}_V = \mathbb{E}\big[\rho_{\tau,2}(Q_{\bar\phi}(s_t,{a}_t) - V(s_t))\big]
```
```math
y = {r}_t + \gamma^H V(s_{t+H}), \qquad
\mathcal{L}_Q = \mathbb{E}\big[(Q(s_t,{a}_t) - y)^2\big]
```
### 3.4 小例子
状态 `s'` 上数据集有 3 个动作，当前 `Q` 为：
```math
Q(s',a_1)=0.1,\quad Q(s',a_2)=0.2,\quad Q(s',a_3)=0.9
```
| 方法 | 结果 | 风险 / 特点 |
|------|------|-------------|
| `\max_{a'} Q` | `0.9`；若对未见 `a_4` 乱估可能到 `1.5` | OOD 过估计 |
| 普通均值 `V` | `0.4` | 抹掉高回报模式 `a_3` |
| IQL expectile（`\tau>0.5`） | 介于均值与 max 之间，更靠近 `0.9`，仍由这三个点决定 | **相对安全的乐观** |
### 3.5 IQL 训练流程示意
```text
离线数据 (s, a, r, s')
        │
        ▼
┌───────────────────────────────┐
│ 用数据动作的 Q̄ 值              │
│ 通过 expectile (τ > 0.5)      │
│ 拟合 V(s)                     │  ← 隐式偏好「数据里较好动作」
└───────────────┬───────────────┘
                │
                ▼
          y = r + γ V(s')         ← 没有 max_a' Q
                │
                ▼
          用 y 训练 Q(s, a)
                │
                ▼
     （可选）策略提取：更偏向高 Q 的动作
     例如 advantage-weighted regression
```
### 3.6 IQL 的优点与局限
**优点**
- 适合离线 / 异策略大数据
- 避免显式 `\max` 带来的 OOD 过估计
- 能利用失败轨迹上的 reward（不只模仿成功演示）
**局限**
- `V(s)` 是**单个标量**；若同一状态上回报**多峰、重尾**（成功 / 失败 / 干预混杂），一个 expectile 仍可能压扁结构
- 机群级异构数据下，往往需要保留整幅价值分布，而不仅是一个偏高平均值
→ 引出 DIVL。
---
## 4. DIVL：Distributional Implicit Value Learning
> 来源：LWD (*Learning while Deploying*) 的价值学习模块。  
> 一句话：**保留 IQL 的 in-support 不对称乐观 bootstrap，但用「分布 + 分位数」代替「标量 expectile `V`」。**
### 4.1 动机
机群部署数据常见特征：
- 多任务、多场景、多策略版本异步混入
- 稀疏奖励、失败 / 恢复 / 人类干预并存
- 同一 `(s)` 上回报往往**多峰、重尾**
标量 `V` 容易把稀有但可复现的高回报「平均掉」。DIVL 改为：
1. 先学习「该状态下数据集动作 `Q` 值」的**分布**
2. 再取该分布的 `\tau`-**分位数**作为 bootstrap 统计量
3. （可选）用分布熵自适应调节 `\tau` 的乐观程度
### 4.2 `V_\psi` 学的是什么分布
```math
p_\psi(v \mid s_t) = P\!\big(v = Q_\phi(s_t, {a}_t) \;\big|\; {a}_t \sim \mathcal{D}(\cdot \mid s_t)\big)
```
| 符号 | 含义 |
|------|------|
| `{a}_t \sim \mathcal{D}(\cdot \mid s_t)` | replay 中条件于 `s_t` 的经验动作分布 |
| `Q_\phi(s_t,{a}_t)` | critic 给这些动作的打分 |
| `p_\psi(v \mid s)` | 这些分数的推前分布（pushforward） |
注意：这里的 `V_\psi` **不是**「最优价值」，而是「**数据集行为下 `Q` 值直方图**」的参数化模型。
实现上常用 categorical 分布（C51 风格）：固定支撑 `\{V_i\}_{i=1}^{K}`，网络输出 logits → softmax。LWD 中示例为 `K=201`，支撑约 `[-0.1, 1.1]`。
### 4.3 拟合分布：式 (似然 / 交叉熵)
用 EMA critic 的标量输出作监督：
```math
\mathcal{L}_V(\psi) = \mathbb{E}_{(s_t,{a}_t)\sim\mathcal{D}}\Big[-\log p_\psi\big(Q_{\bar\phi}(s_t,{a}_t) \mid s_t\big)\Big]
```
实现细节：将 `Q_{\bar\phi}` clip 到支撑上，线性投影到相邻 atom，得到目标分布 `m`，再交叉熵训练（C51 projection）。
**内涵：** `V_\psi` 学习「该状态下历史动作好不好、好到什么程度」的整幅分布，而不是一个平均值。
### 4.4 分位数 bootstrap → 训练 `Q`
`\tau`-分位数：
```math
\mathrm{Quant}_\tau\big(V_\psi(s)\big) = \inf\big\{v : F_\psi(v \mid s) \ge \tau\big\}
```
其中 `F_\psi` 为 CDF。`\tau=0.5` 为中位数；`\tau` 越大越乐观。
TD 目标与 critic 损失：
```math
y_Q = {r}_t + \gamma^H \, \mathrm{Quant}_\tau\big(V_\psi(s_{t+H})\big)
```
```math
\mathcal{L}_Q(\phi) = \mathbb{E}\big[\big(Q_\phi(s_t,{a}_t) - y_Q\big)^2\big]
```
| 项 | 含义 |
|----|------|
| `{r}_t` | 当前 chunk 的真实折扣奖励（稀疏设定下多数为 0） |
| `\gamma^H \mathrm{Quant}_\tau(V(s'))` | 「下一状态起，按数据中偏高价值动作继续，大概能拿多少」 |
| 不用 `\max_{{a}} Q` | 避免对未见动作乱估高 |
**与 IQL 的对应关系：**
- IQL：`V` 本身就是 expectile 标量，直接进入 `y`
- DIVL：`V` 是分布，取出分位数后再进入 `y`
- 哲学相同：**in-support 的不对称乐观**
### 4.5 统一视角：非对称损失族
```math
\rho_{\tau,p}(u) = \lvert \tau - \mathbb{I}(u < 0) \rvert \cdot \lvert u \rvert^p
```
| `p` | 统计量 | 方法 |
|------|--------|------|
| `p=2` | expectile | IQL |
| `p=1` | quantile | DIVL |
命题（理想极限下）：直接对数据集 `Q` 值做非对称回归，与「先拟合分布再提取对应统计量」具有相同最优标量解。因此 DIVL 改变的是**表示与可扩展性**（多峰、熵自适应），不是换成另一套乐观原则。
### 4.6 自适应 `\tau`：用不确定性调节乐观程度
归一化熵（`C` 为类别数）：
```math
\mathcal{H}(s) = -\frac{1}{\log C}\sum_{c=1}^{C} p_{\psi,c}(s)\log p_{\psi,c}(s) \in [0,1]
```
| `\mathcal{H}` | 含义 |
|-----------------|------|
| 接近 1 | 分布很平 → 价值估计不确定 |
| 接近 0 | 接近单峰 → 估计较笃定 |
自适应调度：
```math
\tau(s) = \mathrm{clip}\big(\tau_{\mathrm{base}} - \alpha\,\mathcal{H}(s),\; \tau_{\min},\; \tau_{\max}\big)
```
- 不确定 → 降低 `\tau` → 更保守，减轻过估计
- 确定 → 接近 `\tau_{\mathrm{base}}` → 更敢追高价值模式
实践注意：`\tau(s)` 对 TD 目标通常 **stop-gradient**，避免网络靠操纵熵来投机。
LWD 超参示例：离线 `\tau_{\mathrm{base}}=0.6`，在线 `\tau_{\mathrm{base}}=0.9`，`\alpha=0.3`。
### 4.7 `n`-step TD（长时程稀疏奖励）
```math
y_Q = \sum_{i=0}^{n-1}\gamma^{iH}{r}_{t+iH} + \gamma^{nH}\,\mathrm{Quant}_{\tau(s_{t+nH})}\big(V_\psi(s_{t+nH})\big)
```
- 前一项：把未来 `n` 个 chunk 的真实奖励提前写入目标
- 后一项：窗口外仍用分位数 bootstrap
- 中途终止：截断真实回报，并去掉 bootstrap
LWD 中的实践选择：
| 阶段 | `n` | 原因 |
|------|-------|------|
| 离线长任务 | `n=10` | 加速稀疏成功信号回传 |
| 离线短任务 / 在线 | `n=1` | 在线轨迹混有人类干预，长 backup 易污染 TD 路径 |
### 4.8 DIVL 一次更新数据流
```text
batch (s, a, r, s')
        │
        ├─► Q̄(s,a) ──投影成目标分布 m──► 更新 ψ（学分布）
        │
        ├─► 算 H(s') → τ(s') → Quant_τ(Vψ(s'))
        │                      │
        │                      └─► y = r + γ^{…} Quant…
        │
        └─► (Qϕ(s,a) - y)² ──► 更新 ϕ
                 │
                 └─► EMA 更新 Q̄
```
常配合 **clipped double-Q**（两套 `Q` 取 min）进一步抑制过估计。
### 4.9 在 LWD 中的位置（上下文）
LWD 的完整算法 = **DIVL（价值）** + **QAM（策略提取）**。

本笔记覆盖价值侧。系统飞轮、QAM 与机群实验见 [`../vla-rl/LWD.md`](../vla-rl/LWD.md)。
---
## 5. IQL vs DIVL 对照
| 维度 | IQL | DIVL |
|------|-----|------|
| `V` 的形式 | 标量 | 状态条件分布 `p_\psi(v\mid s)` |
| 不对称统计量 | expectile（`p=2`） | quantile（`p=1`） |
| TD bootstrap | `y = r + \gamma V(s')` | `y = r + \gamma\,\mathrm{Quant}_\tau(V(s'))` |
| 乐观来源 | `\tau > 1/2` 的 expectile | `\tau`-分位数；可再加熵自适应 `\tau(s)` |
| 是否显式 `\max_a Q` | 否 | 否 |
| 多峰回报 | 易被压成单值 | 可保留高回报模态 |
| 不确定性利用 | 通常固定 `\tau` | 可用 `\mathcal{H}(s)` 调节 `\tau` |
| 典型场景 | 通用离线 RL | 异构机群 / 稀疏长时程后训练（如 LWD） |
**共同哲学：**
> 在数据集动作支撑内做不对称乐观价值估计，避免 OOD `\max` 过估计。
**DIVL 相对 IQL 多出来的能力：**
> 分布表示 + 分位数提取 +（可选）不确定性感知的自适应乐观。
---
## 6. 自检问题
1. `V(s)` 和 `Q(s,a)` 的差别是什么？
2. 为什么离线 RL 害怕 `\max_{a'} Q(s',a')`？
3. `\tau > 0.5` 的 expectile 相对均值更乐观还是更悲观？它会用到数据外动作吗？
4. IQL 的「Implicit」到底隐式在哪里？
5. DIVL 的 `V_\psi(s)` 是最优价值吗？若不是，它表示什么？
6. 为什么高熵时要降低 `\tau`？
7. 用一句话说明 IQL 与 DIVL 的相同点与不同点。
---
## 7. 符号速查表
| 符号 | 含义 |
|------|------|
| `s, a, r, s'` | 状态、动作、奖励、下一状态 |
| `\gamma` | 折扣因子 |
| `\mathcal{D}` / `\mathcal{B}` | 数据分布 / replay buffer |
| `Q_\phi(s,a)` | 动作价值（critic） |
| `Q_{\bar\phi}` | critic 的 EMA target |
| `V_\psi(s)` | IQL：标量状态价值；DIVL：价值**分布**模型 |
| `\tau` | 不对称程度 / 分位水平；越大越乐观 |
| `\rho_{\tau,p}` | 非对称 `L_p` 损失；`p=2` expectile，`p=1` quantile |
| `\mathrm{Quant}_\tau` | 分布的 `\tau`-分位数 |
| `\mathcal{H}(s)` | 价值分布的归一化熵（不确定性） |
| `{a}_t, H` | action chunk 及其长度 |
| `{r}_t` | chunk 内折扣奖励和 |
---
## 参考文献
1. Ilya Kostrikov, Ashvin Nair, Sergey Levine. **Offline Reinforcement Learning with Implicit Q-Learning**. arXiv:2110.06169, 2021.
2. Marc G. Bellemare, Will Dabney, Rémi Munos. **A Distributional Perspective on Reinforcement Learning**. ICML, 2017.（categorical / C51）
3. Yi Wang et al. **Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies**. 2026.（DIVL 提出与机群实验）
---
*笔记用途：个人知识储备；公式编号与论文不完全一一对应，以理解为优先。*
