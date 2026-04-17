# π0.6 到 π0.6\*：基于 Advantage Conditioning 的训练全流程整理

> 这份文档基于论文相关章节的阅读整理，目标是把 **π0.6 是怎么做“强化学习”** 这件事讲清楚。  
> 重点不在复现所有论文细节，而在于把方法链路串起来：**问题 → 思想 → 目标函数 → 实现方式 → pre-train → post-train**。  
> 术语上，本文把当前行为策略/参考策略统称为 `π_ref`；在具体章节里，`π0.6` 可理解为一个当前版本的 VLA policy，而 `π0.6*` 是 advantage-conditioned 之后的改进策略。

---

## 1. 一句话总结

这套方法**不是**传统 PPO / REINFORCE 式的 policy gradient 强化学习。  
它更接近一种：

- **用 reward / return 训练 value critic**
- **用 critic 计算 advantage**
- **把 advantage 转成“正/负”的条件信号**
- **再把 policy 训练改写成条件行为克隆 / 条件生成**

因此，它本质上是：

> **critic 驱动的 policy extraction / advantage-conditioned policy optimization**

也可以理解成：

> 用 RL 的评价信号，去构造一个更稳定的监督学习目标。

---

## 2. 问题：为什么不能直接做普通 RL

在机器人 VLA（Vision-Language-Action）训练中，直接做标准 RL 会遇到几个问题：

### 2.1 数据分布受限
训练数据往往来自：

- 人类 demonstration
- 历史版本策略 rollout
- 当前策略自主尝试

因此数据天然是一个 **behavior policy** 诱导的分布，而不是任意新策略都能稳定覆盖的分布。

### 2.2 直接 policy gradient 不稳定
对于大模型尤其是混合动作（离散 + 连续）输出的 VLA：

- 直接做 PPO / actor-critic 代价高
- on-policy 更新昂贵
- reward 噪声大
- 连续动作 likelihood 难精确建模
- 大模型在 RL 梯度下容易不稳定

### 2.3 需要“保守改进”
目标通常不是让策略一下子偏离旧策略很远，而是：

> **在保持接近参考策略的同时，优先提升那些高质量动作的概率。**

---

## 3. 核心思想：Regularized RL + Advantage-Conditioned Policy Extraction

这套方法背后的思想可以分成两层。

---

### 3.1 第一层：Regularized RL

普通 RL 想最大化：

```math
J(\pi)=\mathbb{E}_{\tau\sim \rho_\pi}[R(\tau)]
```

其中一条轨迹的 return 定义为：

```math
R(\tau)=\sum_{t=0}^{T} r_t
```

但在实际训练中，策略不希望偏离参考策略 `π_ref` 太远，因此使用 regularized RL：

```math
J(\pi,\pi_{\text{ref}})
=
\mathbb{E}_{\tau\sim\rho_\pi}\left[\sum_{t=0}^T \gamma^t r_t\right]
-
\beta\,\mathbb{E}_{o\sim\rho_\pi}\left[D\!\left(\pi(\cdot|o)\,\|\,\pi_{\text{ref}}(\cdot|o)\right)\right]
```

含义：

- 第一项：希望 reward 高
- 第二项：惩罚新策略偏离参考策略
- `β`：控制“激进改进”与“保守跟随”之间的权衡

如果 `D` 取 KL divergence，会得到经典结果：

```math
\hat{\pi}(a|o)\propto \pi_{\text{ref}}(a|o)\exp\left(\frac{A^{\pi_{\text{ref}}}(o,a)}{\beta}\right)
```

这说明：

> **新策略 = 旧策略 × 对高 advantage 动作的重加权**

---

### 3.2 第二层：把“按 advantage 重加权”变成条件生成

论文进一步引入 improvement indicator `I`，表示某个动作是否是“改进动作”。

它把“改进后的策略”改写成：

```math
\hat{\pi}(a|o,\ell)\propto
\pi_{\text{ref}}(a|o,\ell)
\left(
\frac{\pi_{\text{ref}}(a|I,o,\ell)}{\pi_{\text{ref}}(a|o,\ell)}
\right)^{\beta}
```

特别地，当 `β = 1` 时：

```math
\hat{\pi}(a|o,\ell)=\pi_{\text{ref}}(a|I,o,\ell)
```

这件事非常关键。它意味着：

> 如果我们能训练一个 policy 同时表示  
> `π(a|o,ℓ)` 和 `π(a|I,o,ℓ)`，  
> 那么就能通过条件 `I` 提取出改进策略。

这就是整套方法的核心。

---

## 4. 关键概念

---

### 4.1 Return

轨迹总回报：

```math
R(\tau)=\sum_{t=0}^{T} r_t
```

如果从某个时间步 `t` 开始看未来 return，则记为：

```math
R_t(\tau)=\sum_{t'=t}^{T} r_{t'}
```

---

### 4.2 Value

在给定当前 observation / state 和任务条件下，未来累计回报的期望：

```math
V^\pi(o_t,\ell)=\mathbb{E}\left[\sum_{t'=t}^{T} r_{t'}\mid o_t,\ell\right]
```

直觉上：

- `return` 是**一条具体轨迹**上的实际累计奖励
- `value` 是**从当前状态出发**未来回报的期望

---

### 4.3 Advantage

动作相对于当前状态平均水平的“好多少”：

```math
A^\pi(o_t,a_t,\ell)=Q^\pi(o_t,a_t,\ell)-V^\pi(o_t,\ell)
```

文中使用的是基于 n-step / Monte Carlo return 的估计形式，本质仍然是：

> **advantage = 动作价值 - 状态基线价值**

如果某个动作后续带来的累计回报高于状态平均水平，则 advantage 为正。

---

## 5. 整体解决方案：π0.6 如何做“强化学习”

这套方法可以理解为一个循环：

1. 用当前 policy（例如 `π0.6`）收集数据
2. 依据轨迹 reward 训练 value function
3. 用 value 计算每个动作的 advantage
4. 将 advantage 离散为 improvement label
5. 用该 label 作为额外条件，训练 policy
6. 推理时启用“positive advantage”条件，得到改进后的 `π0.6*`

注意：

> actor 的训练形式是监督学习，  
> 但监督信号来自 reward/value/advantage，  
> 所以整体仍然属于 RL 驱动的 policy improvement。

---

## 6. 实现细节一：Distributional Value Function Training

论文不是直接回归一个标量 `V(o_t,\ell)`，而是先预测一个 **value distribution**：

```math
p_\phi(V \mid o_t,\ell)\in \Delta_B
```

也就是说，模型输出的是一个在 `B` 个离散 bins 上的概率分布。

文中使用：

- `B = 201` 个 value bins

### 6.1 bin 是什么
可以把连续的 value / return 轴切成很多小区间，每个区间就是一个 bin。  
模型预测的是：

> “未来 return 落在第 1 个 bin、第 2 个 bin、……、第 201 个 bin 的概率”

### 6.2 监督标签怎么来
对数据集中的每条轨迹 `τ`、每个时间步 `t`：

```math
R_t(\tau)=\sum_{t'=t}^{T} r_{t'}
```

然后把 `R_t(\tau)` 离散化成某个 bin，记为 `R_t^B(\tau)`。

### 6.3 训练目标
最小化交叉熵：

```math
\min_\phi
\mathbb{E}_{\tau\in \mathcal{D}}
\left[
\sum_{o_t\in\tau}
H\big(R_t^B(\tau),\,p_\phi(V|o_t,\ell)\big)
\right]
```

### 6.4 为什么这样做
好处包括：

- 分类损失通常比直接回归更稳
- 能表达 value 的分布 / 不确定性
- 对多任务和不同 reward 尺度更鲁棒

### 6.5 如何从 distribution 恢复标量 value
训练完后取分布期望：

```math
V^{\pi_{\text{ref}}}(o_t,\ell)
=
\sum_b p_\phi(V=b|o_t,\ell)\,v(b)
```

其中 `v(b)` 是第 `b` 个 bin 对应的代表值。

---

## 7. 实现细节二：用 value 计算 advantage 并构造标签

有了 `V^{π_ref}` 后，就可以对每个数据样本 `(o_t,a_t,\ell)` 估计：

```math
A^{\pi_{\text{ref}}}(o_t,a_t,\ell)
```

然后根据任务阈值 `\epsilon_\ell` 构造 improvement indicator：

```math
I_t=\mathbf{1}\big(A^{\pi_{\text{ref}}}(o_t,a_t,\ell)>\epsilon_\ell\big)
```

也就是：

- `I_t = 1`：该动作 advantage 高于阈值，是“改进动作”
- `I_t = 0`：不是改进动作

论文中进一步把它实现成文本条件：

- `I_t = 1` → `"Advantage: positive"`
- `I_t = 0` → `"Advantage: negative"`

这一步很重要，因为它把 RL 评价信号转成了 VLA 最擅长处理的 **语言条件**。

---

## 8. 实现细节三：Policy 是怎么训练的

这是整套方法最关键的地方。

### 8.1 抽象目标

论文给出的 policy 目标可以写成：

```math
\min_\theta
\mathbb{E}_{\mathcal{D}_{\pi_{\text{ref}}}}
\left[
-\log \pi_\theta(a_t|o_t,\ell)
-\alpha \log \pi_\theta(a_t|I_t,o_t,\ell)
\right]
```

这表示 policy 同时学习两种分布：

1. **普通动作分布** `π(a|o,ℓ)`
2. **improvement-conditioned 动作分布** `π(a|I,o,ℓ)`

其中：

- 第一项：标准行为克隆，让模型复现数据中的一般动作
- 第二项：让模型在给定 improvement 标签时，也能复现动作

所以它的实质是：

> **同一个模型同时学习“普通模式”和“按优势条件控制的模式”**

---

### 8.2 为什么这和 classifier-free guidance 类似

论文提到它与 classifier-free guidance 的原则相似。

这里的相似点不是说它直接使用了扩散模型里的 CFG 公式，而是：

- CFG：同一个模型学“带条件”和“不带条件”的生成
- 这里：同一个模型学“普通动作分布”和“优势条件动作分布”

因此，它共享一种训练原则：

> **把一个额外条件显式喂给生成模型，使其能切换分布。**

---

## 9. 在 VLA 里的具体落地：Advantage Conditioning

论文将 improvement indicator 作为一个额外文本输入加入模型序列：

- 在语言指令 `ℓ` 之后
- 在动作 token 之前

例如：

```text
Task: pick up the cup
Advantage: positive
[actions...]
```

或：

```text
Task: pick up the cup
Advantage: negative
[actions...]
```

这样设计的目的，是只让这个条件影响后续动作生成，而不改变视觉观测和任务本身的定义。

---

## 10. 动作为什么分成离散部分和连续部分

VLA policy 的动作表示通常是混合的：

- **离散动作序列**
- **连续控制量**

因此，policy 的训练也分成两部分。

### 10.1 离散动作部分
可以直接计算 token log-likelihood，即标准 next-token / cross-entropy 训练。

### 10.2 连续动作部分
连续部分的 log-likelihood 不能直接精确求解，因此论文采用 **flow matching loss** 来训练连续动作生成器。

---

## 11. Continuous Action：Flow Matching 是怎么介入的

论文把连续动作训练写成一个 flow matching 目标。直观上，它做的是：

- 取真实动作 `a`
- 采样高斯噪声 `ω ~ N(0, I)`
- 构造一个噪声动作与真实动作之间的插值点

```math
a^{\eta,\omega}_{t:t+H}=\eta a_{t:t+H} + (1-\eta)\omega,\quad \eta\in[0,1]
```

其中：

- `η = 0` 时接近纯噪声
- `η = 1` 时是干净动作

模型 `f_\theta` 在给定：

- noisy action
- observation `o_t`
- task language `ℓ`
- improvement indicator `I_t`

的情况下，学习预测将 noisy action 恢复为真实 action 的方向。

因此，整体上可以把 action likelihood 的优化近似写成：

- 离散动作：最大化 log-likelihood
- 连续动作：最小化 flow matching loss

于是，policy 的实际训练变成：

> **离散动作做语言模型式监督，连续动作做扩散/流模型式监督**，  
> 且两者都条件在 `Advantage: positive / negative` 上。

---

## 12. 训练循环：从 π0.6 到 π0.6\*

下面用一个完整流程串起来。

### Step 1：初始化 policy
从一个已有 VLA policy 开始，例如 `π0.6`。

### Step 2：收集数据
用当前 policy 执行任务，得到轨迹数据集：

```math
\mathcal{D}_{\pi_{0.6}}
```

数据通常包括：

- human demonstrations
- autonomous rollouts
- 也可能包括历史轮次数据

### Step 3：计算轨迹 return
对每条轨迹 `τ` 的每个时间步 `t` 计算：

```math
R_t(\tau)=\sum_{t'=t}^{T}r_{t'}
```

### Step 4：训练 distributional value function
利用 `(o_t, ℓ) -> R_t(\tau)` 作为监督，训练 critic：

```math
p_\phi(V|o_t,\ell)
```

并取期望得到连续值：

```math
V^{\pi_{\text{ref}}}(o_t,\ell)
```

### Step 5：计算 advantage
对每个动作 `(o_t,a_t,\ell)` 计算：

```math
A^{\pi_{\text{ref}}}(o_t,a_t,\ell)
```

### Step 6：构造 improvement 标签
依据阈值生成：

```math
I_t=\mathbf{1}(A^{\pi_{\text{ref}}}(o_t,a_t,\ell)>\epsilon_\ell)
```

然后转为文本：

- `"Advantage: positive"`
- `"Advantage: negative"`

### Step 7：训练 policy
让 policy 在两种条件下都拟合动作：

- `π(a|o,ℓ)`：普通动作分布
- `π(a|I,o,ℓ)`：优势条件动作分布

训练目标由两部分构成：

- 离散动作 NLL
- 连续动作 flow matching loss

### Step 8：得到改进策略
推理时，把条件设为：

```text
Advantage: positive
```

则模型更倾向生成训练中被 critic 判定为高 advantage 的动作，从而得到 `π0.6*`。

---

## 13. 伪代码版本

```python
# initialize
policy = pi_0_6
value_model = V_phi

for iteration in range(num_iters):
    # 1) collect data
    D = collect_trajectories(policy, demos=True, autonomous=True)

    # 2) compute Monte Carlo returns
    for tau in D:
        for t in reversed(range(len(tau))):
            tau[t].return_to_go = sum_reward_from_t_to_end(tau, t)

    # 3) train distributional value function
    train_value_model(
        inputs=[(o_t, l_t) for all samples],
        targets=[discretize(return_to_go, B=201)]
    )

    # 4) estimate advantage
    for sample in D:
        sample.value = value_model.expectation(sample.obs, sample.lang)
        sample.advantage = estimate_advantage(sample, value_model)

    # 5) build improvement label
    for sample in D:
        sample.I = int(sample.advantage > epsilon(sample.task))
        sample.adv_text = "Advantage: positive" if sample.I else "Advantage: negative"

    # 6) train policy
    train_policy(
        inputs=[
            (obs, lang),
            (obs, lang, adv_text)
        ],
        targets=actions,
        loss = discrete_action_nll + continuous_action_flow_matching
    )

# inference
action = policy.sample(obs, lang, adv_text="Advantage: positive")
```

---

## 14. Pre-train 在做什么

根据论文上下文，pre-train 阶段重点是：

### 14.1 数据来源
主要来自 **human demonstrations**。

### 14.2 Value 学到什么
此时 value function 近似刻画的是：

```math
V^{\pi_{\text{ref}}}
```

而这里的 `π_ref` 主要对应 demonstration policy / 人类行为分布。

### 14.3 Policy 学到什么
policy 在 pre-train 时主要是在 demonstration 数据上做：

- 普通行为建模
- 基于 critic 的 positive / negative advantage conditioning

可以把它理解为：

> 先让模型学会在已有高质量数据分布上，识别并偏向更优动作。

### 14.4 pre-train 的意义
- 建立一个可靠的多任务 critic
- 给 policy 一个稳定的初始行为分布
- 让 advantage conditioning 先在 demonstration 上生效

---

## 15. Post-train 在做什么

post-train 阶段不再只依赖 demonstrations，而是加入模型自主尝试数据。

### 15.1 数据集变化
数据集 `D` 开始混合：

- demonstrations
- 当前 / 历史 policy 的 autonomous attempts

因此此时的 `π_ref` 不再只是人类行为，而是一个混合参考分布。

### 15.2 Critic 的变化
value function 此时学到的是：

> demonstration return 与 learned policy return 的加权混合下的 value

它不再只评估人类分布，而是逐渐贴近当前训练生态中的真实数据分布。

### 15.3 Policy 的变化
policy 继续根据 advantage 标签学习：

- 哪些自主尝试动作优于当前平均水平
- 哪些 demonstrations 仍然最优
- 如何在多源数据中提取“更值得模仿”的动作模式

### 15.4 post-train 的本质
post-train 可以看成一个 **迭代的自举过程（bootstrapping）**：

1. 当前 policy 收集数据
2. critic 重新估计 value / advantage
3. policy 在 positive advantage 条件下继续拟合更好的动作
4. 策略得到进一步提升

---

## 16. 为什么这种方法有效

这套方法的优势主要来自下面几点。

### 16.1 避免直接 policy gradient 的不稳定
actor 的训练是监督式的：

- 更稳定
- 更适配大模型
- 更容易与语言/视觉输入统一建模

### 16.2 把 RL 信号转成条件生成
reward -> return -> value -> advantage -> positive/negative token

这条链路把原本难优化的 RL 信号，转成了模型更容易吸收的监督标签。

### 16.3 兼容混合动作空间
离散动作部分用 NLL，连续动作部分用 flow matching，适合 VLA 的动作表示。

### 16.4 可以做保守改进
策略不是完全抛弃旧策略，而是在旧行为分布上提升高 advantage 动作的概率，更符合离线 / 批量 RL 的约束。

---

## 17. 它和 PPO / SAC / AWR / AWAC / IQL 的关系

### 17.1 不像 PPO / SAC
它**不是**直接最大化 actor 的 expected return，也没有显式的 policy gradient / importance ratio / entropy regularization 的标准在线 RL 结构。

### 17.2 更像 AWR / AWAC
它和 AWR / AWAC 的共同点在于：

- 都使用 critic / advantage 去指导 actor
- 都不是纯 imitation learning
- 都在“高 advantage 动作更值得模仿”的方向上更新 policy

### 17.3 和 IQL 也有亲缘关系
相似点在于：

- 利用 value / advantage 做隐式 policy improvement
- actor 更新采用 supervised / weighted regression 风格，而非直接 RL 梯度

### 17.4 这篇方法的独特点
它把 advantage 信息显式编码成：

- 一个 **语言条件**
- 一个 **生成控制信号**

使得同一个 VLA 模型可以在推理时通过条件切换行为模式。

---

## 18. 实践理解：它到底算不算强化学习

算，但不是经典狭义的 policy gradient RL。

更准确的表述应该是：

> **这是一种利用 reward 训练 critic，再通过 critic 引导 policy 进行监督式提取的 RL 方法。**

换句话说：

- actor 的更新形式是 supervised learning
- actor 的监督标签来自 RL 信号
- 因此整体仍然属于 RL 驱动的 policy optimization

---

## 19. 最终结论

`π0.6` 这套“强化学习”流程，本质上可以概括为：

> **先用轨迹 reward 训练一个可靠的 value critic；再用 critic 给每个动作打 advantage 分数；再把“高 advantage / 低 advantage”转成额外语言条件；最后用条件行为克隆 + flow matching 训练 policy。**

因此，从 `π0.6` 到 `π0.6*` 的关键，不是直接套用 policy gradient，而是：

> **把 RL 的改进信号蒸馏成一个可控的条件生成目标。**
