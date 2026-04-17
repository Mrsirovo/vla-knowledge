# TGRPO：通过 Trajectory-wise Group Relative Policy Optimization 微调 VLA

> 论文：**TGRPO: Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization**  
> 这份文档按“问题 → 思想 → 方法 → 训练公式 → 实现 → 实验结论”的顺序整理。  
> 重点放在 **训练机制**，尤其是：
>
> 1. 为什么作者要把 GRPO 改成适合机器人长时序任务的版本  
> 2. 多阶段 dense reward 是怎么构造的  
> 3. step-level 与 trajectory-level advantage 分别在解决什么问题  
> 4. 最终的 TGRPO 损失函数到底在优化什么  
>
> 说明：论文 HTML 版本对部分公式的字符解析不完整，下面的公式根据正文定义和上下文重建为 **等价、可实现且适合 GitHub 渲染** 的形式。

---

# 1. 论文要解决什么问题

这篇论文关注的是：**如何用在线强化学习微调 Vision-Language-Action (VLA) 模型**，使其在新任务或分布外场景中比纯 SFT 更鲁棒、更能利用环境反馈。

作者认为，现有 VLA 微调存在三个核心问题：

## 1.1 只依赖成功 demonstration，无法从失败中学习

SFT 的基本训练数据通常只有成功轨迹，因此模型学到的是“动作模仿”，而不是“根据环境反馈持续修正行为”。  
一旦进入新环境、出现执行偏差，SFT policy 的恢复能力较弱。

## 1.2 长时序机器人任务 reward 稀疏，直接做 RL 很难

很多操作任务只有最终成功/失败信号，如果直接用终局二值 reward 做 RL，会遇到：

- credit assignment 困难
- advantage 方差大
- 训练不稳定
- 长时序下很容易陷入局部最优

## 1.3 单一粒度的 advantage 不够

如果只看整条轨迹的总回报，那么能学到“这一整条轨迹好不好”，但不知道具体哪一步动作起了作用。  
如果只看每个时间步的局部 reward，又容易丢失整条轨迹最终是否成功的全局信息。

因此，作者的核心主张是：

> 对 VLA 做在线 RL 时，需要同时利用 **局部步骤信息** 和 **整条轨迹信息**。

---

# 2. TGRPO 的核心思想

TGRPO 的名字可以直接拆开理解：

- **Trajectory-wise**：显式利用整条轨迹层面的相对优劣
- **Group Relative**：不是对单个样本绝对打分，而是在一个 group 内做相对比较
- **Policy Optimization**：最终仍然是策略梯度式更新，但不依赖单独训练的 value network

它继承了 GRPO 的一个基本思想：

> 不训练 critic，而是在同组样本内部做相对归一化，构造 advantage。

但机器人任务和语言任务不一样。  
对于长时序 manipulation，仅做 trajectory-level group comparison 还不够，因为：

- 两条轨迹最后都失败，但前半段质量可能差很多
- 某条轨迹虽然最终失败，但早期动作可能是合理的
- 某条轨迹虽然最终成功，但中途可能存在很多低质量动作

于是作者提出：**把相对 advantage 分成两个层级再融合**。

---

# 3. 方法总览

TGRPO 的完整训练链条可以写成：

 ```math
 \text{观测 + 指令} \rightarrow \text{VLA 采样多条轨迹} \rightarrow \text{LLM 生成的多阶段 dense reward} \rightarrow \text{step-level 分组} + \text{trajectory-level 分组} \rightarrow \text{relative advantage 融合} \rightarrow \text{clipped policy update} 
 ```

具体包括三个组成部分：

## 3.1 多阶段 reward 设计
作者不用纯终局 reward，而是借助 LLM 根据任务描述自动生成多阶段 reward。

## 3.2 双层相对 advantage
- 在**相同时间步**上跨轨迹比较，得到 step-level relative advantage  
- 在**整条轨迹**上比较总回报，得到 trajectory-level relative advantage

## 3.3 融合后的策略优化
把这两个 advantage 加权融合，然后放进一个类似 PPO / GRPO 的 clipped surrogate objective 中更新 policy。

---

# 4. 多阶段 reward 设计

这一部分对应论文的方法章节 **IV-A Multi-Stage Reward Design**。

作者指出，若把终局二值 reward 直接传播给所有步骤，会把所有动作看成贡献相同，这在机器人长时序任务中显然不成立。  
例如在一个失败轨迹里，早期动作可能已经正确完成了若干子目标；如果统一给低回报，会损失这些局部有效信息。

因此，作者设计了一个 **multi-stage reward**。

## 4.1 核心思路

作者使用 Claude 3.7 Sonnet 根据自然语言任务描述自动拆分子阶段。  
例如“把番茄酱瓶放进篮子”这个任务，可以分成：

- 接近目标物体
- 抓取目标物体
- 搬运到目标区域
- 放置到篮子里

然后 reward 由两部分构成：

1. **基于关键物体状态的 reward**
2. **基于成功示范末端执行器关键位姿的 shaping reward**

因此，按论文文字定义，单步 reward 可以整理为：

 ```math
 r_t = r_t^{\text{obj}} + r_t^{\text{pose}}
 ```

其中：

- $r_t^{\text{obj}}$：根据关键物体之间的空间关系、阶段完成情况等给出的任务 reward
- $r_t^{\text{pose}}$：根据机器人当前末端执行器位置与成功示范关键位姿的距离构造的 shaping reward

## 4.2 这一设计的作用

这一步非常关键。它的作用不是简单“加一个 dense reward”，而是同时解决两个问题：

### 第一，降低 reward sparsity
模型不再只能等到任务结束才收到成功/失败信号，而是每个阶段都有局部反馈。

### 第二，改善 credit assignment
如果某一步动作帮助完成了某个子目标，那么它可以立即得到正向回馈，而不是被终局 reward 淹没。

### 第三，提高采样效率
作者还加入了 demonstration pose reference，因此 reward 不只看“任务是否完成”，还看“动作是否在向示范中的成功几何结构靠近”。

---

# 5. TGRPO 的双层 advantage 设计

这是整篇论文最重要的部分，也是它区别于普通 GRPO 的核心。

假设并行采样得到 $N$ 条轨迹，每条轨迹长度为 $T$。  
记第 $i$ 条轨迹第 $t$ 步的 reward 为 $r_t^{(i)}$。

先定义每条轨迹的总回报：

 ```math
 R^{(i)} = \sum_{t=1}^{T} r_t^{(i)} 
 ```

接下来分别构造两种 relative advantage。

---

## 5.1 Step-level relative advantage

作者把 **所有轨迹在同一时间步 $t$ 的样本** 放在同一个 group 中比较。  
直观上，这相当于问：

> 在第 $t$ 步这一局部决策点上，哪条轨迹的当前动作更优？

按论文描述，这个 step-level relative advantage 可写成同组标准化形式：

 ```math
 A_{\text{step}}^{(i,t)} = \frac{r_t^{(i)} - \mu_t}{\sigma_t + \epsilon}
 ```

其中：

- $\mu_t = \frac{1}{N}\sum_{i=1}^{N} r_t^{(i)}$：第 $t$ 步 across trajectories 的平均 reward
- $\sigma_t$：对应标准差
- $\epsilon$：数值稳定项

### 这个量在优化什么

它强调的是**局部动作质量**。

如果在相同时间步 $t$，第 $i$ 条轨迹的动作比同组其他轨迹更有利于当前阶段推进，那么：

- $r_t^{(i)}$ 更大
- $A_{\text{step}}^{(i,t)}$ 为正

因此，step-level advantage 能告诉模型：

> 当前时间步上，什么样的动作更好。

### 为什么它重要

因为长时序 manipulation 中，许多差异是局部出现的。  
同样最终失败的两条轨迹，在第 8 步抓取、在第 15 步接近、在第 30 步放置动作上可能差异很大。  
trajectory-level 总分看不到这些细节，step-level 才能提供细粒度学习信号。

---

## 5.2 Trajectory-level relative advantage

接着，作者把 **整组轨迹** 当成一个 group 比较它们的总回报。  
直观上，这相当于问：

> 从整条轨迹来看，哪条 rollout 更成功、更高效、更符合任务目标？

trajectory-level relative advantage 可整理为：

 ```math
 A_{\text{traj}}^{(i)} = \frac{R^{(i)} - \mu_R}{\sigma_R + \epsilon}
 ```

其中：

- $\mu_R = \frac{1}{N}\sum_{i=1}^{N} R^{(i)}$
- $\sigma_R$：总回报的标准差

### 这个量在优化什么

它强调的是**全局任务完成质量**。

如果某条轨迹整体上做得更好，无论是更快完成更多子目标，还是最终获得更高总回报，它都会有更大的 $A_{\text{traj}}^{(i)}$。

### 为什么不能只用 step-level

因为机器人任务不是每一步都能独立评价。  
某一步“看起来局部正确”，并不代表它会导向更好的全局结果。  
例如：

- 提前接近了错误的物体
- 短期 reward 增加，但最终无法完成任务
- 局部动作看似合理，但导致后续姿态不可恢复

trajectory-level advantage 可以把这些全局后果带回来。

---

## 5.3 双层 advantage 融合

有了局部与全局两个量之后，作者将它们加权融合，得到每个时间步最终使用的相对 advantage：

 ```math
 A^{(i,t)} = \alpha \, A_{\text{step}}^{(i,t)} + \beta \, A_{\text{traj}}^{(i)}
 ```

其中：

- $\alpha$：step-level 权重
- $\beta$：trajectory-level 权重

### 这个融合在解决什么

这一步非常关键。它在解决一个典型的机器人 RL 矛盾：

- 只看局部：容易短视，只优化“当前这一步”
- 只看全局：credit assignment 粗糙，不知道是哪一步起作用

融合后，每一步更新都同时携带：

- **局部动作质量信息**
- **整条轨迹优劣信息**

因此，这个 advantage 既不会完全丢掉短时控制信号，也不会脱离全局任务目标。

---

# 6. TGRPO 的策略更新目标

有了最终 advantage 后，作者把它放入一个类似 PPO / GRPO 的 clipped objective 中。

先定义 importance sampling ratio：

 ```math
 \rho_t^{(i)}(\theta) = \frac{\pi_\theta(a_t^{(i)} \mid o_t^{(i)}, l)}{\pi_{\theta_{\text{old}}}(a_t^{(i)} \mid o_t^{(i)}, l)}
 ```

这里：

- $\pi_\theta$：当前待更新策略
- $\pi_{\theta_{\text{old}}}$：采样轨迹时使用的旧策略
- $a_t^{(i)}$：第 $i$ 条轨迹第 $t$ 步的动作

接着，策略目标写成：

 ```math
\mathcal{L}_{\text{TGRPO}}(\theta)
=
\mathbb{E}_{i,t}
\left[
\min\left(
\rho_t^{(i)}(\theta) A^{(i,t)},
\;
clip\!\big(\rho_t^{(i)}(\theta), 1-\varepsilon, 1+\varepsilon\big) A^{(i,t)}
\right)
\right]
-
\lambda \, D_{\mathrm{KL}}
\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
 ```

其中：

- $\varepsilon$：clipping threshold
- $\lambda$：KL 正则系数
- $\pi_{\mathrm{ref}}$：参考策略，通常是初始模型或前一阶段模型

### 这个目标的含义

第一部分是 PPO/GRPO 风格的 clipped surrogate：

- 如果新策略提高了高 advantage 动作概率，目标会上升；
- 但如果 ratio 偏移过大，会被 clip 截断，避免更新过猛。

第二部分是 KL 正则：

- 防止策略过快偏离参考模型
- 保持 VLA 的原有表征与基础能力不被 RL 破坏得太厉害

---

## 6.1 为什么说它是 critic-free

因为这里的 advantage **不是来自一个单独学习的 value network**，而是来自同组轨迹内部的相对归一化比较。  
这就是它和 PPO 的一个根本区别：

- PPO：通常需要 value function + GAE
- TGRPO：不用额外训练 critic，而是依靠 group relative normalization 构造 advantage

### 好处

这样做可以减少一个额外价值网络的训练负担，也降低 value estimation 误差对 policy learning 的影响。

### 代价

代价是：这种方法需要同组样本足够有比较意义。  
也就是说，并行环境需要设计得合理，group size 不能太小，否则 relative normalization 不稳定。

---

# 7. 论文中的 KL 项

论文提到，对于 KL divergence 他们采用了一个 unbiased estimator。  
结合 GRPO 一类方法的常见写法，可以整理为如下 sample-wise estimator：

 ```math
D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})
\approx
\frac{\pi_{\mathrm{ref}}(a_t \mid o_t,l)}{\pi_\theta(a_t \mid o_t,l)}
-
\log \frac{\pi_{\mathrm{ref}}(a_t \mid o_t,l)}{\pi_\theta(a_t \mid o_t,l)}
-
1
 ```

### 这一项的作用

这是一种对 reference policy 的约束。  
在 VLA 后训练里，这个约束很常见，因为纯 RL 很容易让模型：

- 遗忘原先的通用先验
- 产生过激的动作分布漂移
- 在 reward shaping 下过拟合局部策略

KL 项相当于在说：

> 允许策略朝更优方向更新，但不要偏离基础模型太远。

---

# 8. TGRPO 的完整训练过程

把上面的所有部分串起来，TGRPO 的训练流程可以写成下面这套算法。

## Step 1：选择一个任务并构造并行环境

作者在 LIBERO 上训练时，对单个任务建立多个并行环境，并让这些环境以**相同初始状态**启动。  
这样可以保证：

- 不同 rollout 之间具有可比性
- 同一时间步的样本能组成 step-level group

## Step 2：用 VLA 在并行环境中采样轨迹

输入为：

- 当前图像观测
- 语言任务指令

输出为：

- 动作 token / action sequence

并执行得到多条轨迹。

## Step 3：根据 LLM 生成的多阶段 reward 计算每步奖励

reward 由：

- 关键物体状态变化
- 与成功示范关键位姿的距离

共同构成，从而得到每步的 dense reward。

## Step 4：计算 trajectory return

对每条轨迹：

 ```math
 R^{(i)} = \sum_{t=1}^{T} r_t^{(i)}
 ```

## Step 5：计算 step-level relative advantage

对每个时间步 $t$，把所有轨迹在该步的 reward 放在一起做标准化：

 ```math
 A_{\text{step}}^{(i,t)} = \frac{r_t^{(i)} - \mu_t}{\sigma_t + \epsilon}
 ```

## Step 6：计算 trajectory-level relative advantage

把整条轨迹的回报放在一起比较：

 ```math
 A_{\text{traj}}^{(i)} = \frac{R^{(i)} - \mu_R}{\sigma_R + \epsilon} 
 ```

## Step 7：融合 advantage

 ```math
 A^{(i,t)} = \alpha A_{\text{step}}^{(i,t)} + \beta A_{\text{traj}}^{(i)}
 ```

## Step 8：计算 ratio 与 clipped objective

 ```math
 \rho_t^{(i)}(\theta) = \frac{\pi_\theta(a_t^{(i)} \mid o_t^{(i)}, l)}{\pi_{\theta_{\text{old}}}(a_t^{(i)} \mid o_t^{(i)}, l)} 
 ```

并最大化：

 ```math
\mathcal{L}_{\text{TGRPO}}(\theta)
=
\mathbb{E}_{i,t}
\left[
\min\left(
\rho_t^{(i)}(\theta) A^{(i,t)},
\;
clip\!\big(\rho_t^{(i)}(\theta), 1-\varepsilon, 1+\varepsilon\big) A^{(i,t)}
\right)
\right]
-
\lambda D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})
 ```

## Step 9：更新 VLA

作者以 OpenVLA 为基础模型，用 LoRA 做 RL 微调。  
也就是说，不是全参数更新，而是低秩适配更新，这样更稳、更省资源。

---

# 9. 这套方法为什么适合机器人 VLA

## 9.1 它解决了 reward sparsity，但不是纯手工 reward engineering

作者借助 LLM 从任务语言中自动拆分子阶段，并结合环境状态生成 reward。  
这比人工逐任务设计 dense reward 的工程成本更低。

## 9.2 它不用额外 critic，但又比纯 trajectory-level GRPO 更细

只用 trajectory-level 比较，会让 credit assignment 太粗。  
只用 step-level 比较，又会丢掉长时序任务的全局成功信号。  
TGRPO 的优势就在于把这两者结合起来。

## 9.3 它天然适配并行仿真

算法需要同组轨迹比较，因此很适合在 LIBERO 这样的 simulator 中用多个并行环境采样。  
这也是它能在线做 RL 微调的实际基础。

---

# 10. 实验结果与论文结论

论文在 LIBERO 四个任务套件上做了实验，结果显示 TGRPO 的平均成功率为 **80.7%**，高于：

- OpenVLA-SFT：76.5%
- OpenVLA-DPO：76.2%
- GRAPE：80.2%
- Octo：73.9%

其中，在更长时序的 LIBERO-Long 上，TGRPO 达到 **59.2%**，优于 SFT 的 **51.1%**。  
这说明双层 advantage 设计在长时序任务上更有价值。 citeturn959098view0turn959098view2

论文在消融实验中还比较了去掉 trajectory-level 或去掉 step-level 的版本。  
在 LIBERO-Goal 上：

- 去掉 trajectory-level 后平均成功率为 80.2%
- 去掉 step-level 后平均成功率为 86.8%
- 完整 TGRPO 为 92.2%

这说明两个层级的 advantage 都有贡献，而且二者结合最好。 citeturn959098view1

---

# 11. 这篇论文最值得记住的点

## 11.1 它不是简单把 GRPO 直接搬到机器人上

机器人操作是长时序控制问题，不是文本 token 生成问题。  
所以作者专门补了一个 **step-level relative advantage**，来修复 trajectory-only 信号过粗的问题。

## 11.2 它的关键不只是 reward 变 dense 了

更重要的是：

> reward 的层级化 + advantage 的双层化 + clipped policy update  
> 三者是一起工作的。

只有 dense reward 而没有合理 advantage 聚合，训练仍然会高方差、不稳定。

## 11.3 它代表了一类新的 VLA 后训练思路

这类方法的共同特点是：

- 先从大规模数据预训练得到通用 VLA
- 再在具体任务上做 RL post-training
- RL 阶段不追求从零学技能，而是让模型在在线交互中学会纠偏、强化任务适配性

TGRPO 就是这一范式下一个比较典型、比较清晰的代表。

---

# 12. 一句话总结

如果只用一句话概括 TGRPO：

> **TGRPO 是一种面向 VLA 在线微调的 critic-free RL 方法，它通过 LLM 生成多阶段 dense reward，再把同时间步的局部相对优势与整条轨迹的全局相对优势加权融合，最终用类似 PPO/GRPO 的 clipped objective 稳定更新策略。**
