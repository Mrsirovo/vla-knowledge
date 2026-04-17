# RoboReward：通用机器人视觉-语言奖励模型

> 论文：**RoboReward: General-Purpose Vision-Language Reward Models for Robotics**  
> 这份文档面向 GitHub 仓库整理，重点放在：
>
> 1. 论文要解决什么问题  
> 2. RoboReward 数据集和基准是怎么构造的  
> 3. 奖励模型具体学什么、怎么训练  
> 4. 为什么作者选择 **离散进度奖励** 而不是二值成功奖励  
> 5. RoboReward 在离线评测和真实机器人 RL 中说明了什么  
>

---

## 1. 一句话概括

RoboReward 的目标是把 **视觉-语言模型（VLM）** 训练成适用于机器人 RL 的**通用奖励模型**。  
核心做法不是直接手工设计 reward，也不是只依赖人工逐条打分，而是：

- 先证明对机器人 RL 来说，**进度型奖励（progress reward）** 比二值成功奖励更适合学习；
- 再把大规模真实机器人数据集中的成功演示，通过 **counterfactual relabeling** 扩展成包含失败和 near-miss 的训练数据；
- 最后在这个数据上训练 3B/7B 的视觉-语言奖励模型，让它输入 **任务文本 + 整段 rollout 视频**，输出 **1 到 5 的离散进度分数**。

作者同时给出一个统一 benchmark，系统评测现有 VLM 作为机器人 reward model 的能力，并展示更准确的 reward model 会带来更好的下游 RL。  
这一点是全文最重要的结论之一。

---

## 2. 论文要解决的问题

机器人 RL 一直有一个老问题：**reward 难定义**。  
在真实机器人场景里，常见做法通常只有两种：

- 人工手写 reward 函数；
- 人工给 episode 打标签。

这两种方式都不够理想。手写 reward 往往 brittle，人工标注又昂贵、慢，而且很难扩展到大量任务与机器人本体。论文的出发点就是：能不能让一个通用的 VLM 直接充当 reward model？

作者认为，现有 VLM 虽然在视觉和语言上很强，但**并不自动意味着它能可靠地给机器人 rollout 打 reward**。  
因为机器人 reward 要求的是：

- 能分辨成功、失败、部分完成；
- 能识别“同一视频对应不同任务文本时”奖励应该不同；
- 对真实机器人长视频、不同 embodiment、不同视角都有一致判断。

而这正是当前通用 VLM 的薄弱点。

---

## 3. 这篇论文的核心贡献

作者的贡献可以概括为四点。

### 3.1 提出 RoboReward 数据集与评测基准
RoboReward 基于 Open X-Embodiment（OXE）和 RoboArena 构建，用于训练和评测通用机器人奖励模型。  
训练集约 64,850 个 episode-reward 对，验证集 2,442 个，测试集 3,105 个；测试集经过人工复核。

### 3.2 提出 counterfactual relabeling
由于 OXE 以成功 demonstration 为主、几乎没有失败样本，作者设计了一套反事实重标注流程，把“同一个成功视频”配上新的任务描述，生成 **negative** 和 **near-miss** 样本。 

### 3.3 训练 RoboReward 3B/7B 奖励模型
作者在整理后的数据上微调 Qwen2.5-VL，训练 3B 和 7B 两个通用 reward model。两者都冻结视觉 backbone，只训练 fusion 和 LLM 层。

### 3.4 证明 reward 质量与下游 RL 表现强相关
作者先在仿真里做了一个重要分析：reward model 的准确度和最终 RL policy 的表现高度相关，相关系数约为 0.83。  
因此，离线 benchmark 不只是“打榜”，而是对实际 RL 有预测意义。

---

## 4. 为什么作者选“离散进度奖励”

这篇论文里最关键的方法选择之一，就是**不把奖励做成二值 success/fail**，而是用 **5 档离散进度分数**。

作者在 Robomimic 仿真分析中比较了三种 reward 形式：

1. Binary success：成功记 1，失败记 0  
2. Continuous progress：连续进度值，范围在 `[0, 1]`  
3. Discrete progress：把进度离散成 5 个等级，奖励在 `{1,2,3,4,5}`

可以写成：

```math
r \in \{0,1\}
```

```math
r \in [0,1]
```

```math
r \in \{1,2,3,4,5\}
```

实验结果表明：

- 学习到的 **binary success reward** 用于 RL 时效果明显差；
- 学习到的 **continuous progress** 和 **discrete progress** 都接近真实 reward 的 RL 效果；
- 离散进度更容易让人稳定标注，因此最终选择 **5 档 progress label** 作为 RoboReward 的目标形式。

这背后的直觉很清楚：

- 成功/失败太粗，难以给长时序任务提供 credit assignment；
- 连续进度虽然细，但人工一致性差；
- 5 档离散进度在“信息量”和“可标注性”之间更平衡。

---

## 5. RoboReward 的监督目标是什么

训练时，模型的输入是：

- 一段机器人 rollout 视频；
- 对应的任务文本描述。

输出是一个 **离散进度标签**，表示这个 rollout 对这个任务完成到了什么程度。

可以写成一个标准的分类问题：

```math
y \in \{1,2,3,4,5\}
```

```math
p_\theta(y \mid v, t)
```

其中：

- `v` 表示整段 rollout 视频；
- `t` 表示任务文本；
- `y` 是 1 到 5 的进度分数。

训练目标本质上就是让模型预测正确的 progress label。  
论文正文没有在你提供的摘要片段里直接给出训练 loss 公式，但从任务定义和模型选择方式看，它就是一个标准的多类分类训练；模型选择时使用的是验证集上的 **MAE（mean absolute error）**。

MAE 可以写成：

```math
MAE = \frac{1}{N}\sum_{i=1}^{N}\left|\hat y_i - y_i\right|
```

这里：

- `y_i` 是真实 progress label；
- `\hat y_i` 是模型预测分数；
- `N` 是样本数。

作者用 held-out validation set 上的 MAE 选择 3B/7B 模型的最佳 checkpoint。

---

## 6. 数据集怎么构造

### 6.1 数据来源

RoboReward 主要来自两个来源：

#### Open X-Embodiment (OXE)
这是一个约 100 万条真实机器人 demonstration 的集合，涵盖 22 种机器人 embodiment 和大量任务。  
但 OXE 的问题是：**几乎全是成功 demonstration**。对于训练 reward model，这种数据分布是不够的，因为 reward model 必须学会同时判断成功、失败与部分完成。

#### RoboArena
RoboArena 是一个更偏评测性质的数据集，包含真实机器人 policy rollout，并且带有人类提供的 progress score。  
它既有成功也有失败，数据分布比 OXE 更适合 reward 建模。作者将 RoboArena 原始 `[0,100]` 进度分数映射到 1 到 5 的离散奖励。

---

## 7. Counterfactual relabeling：这篇论文最有技术含量的部分

因为 OXE 太“成功导向”，作者提出了一个很关键的数据增强机制：**counterfactual relabeling**。

### 7.1 基本思想

给定一个成功视频，不改变视频本身，而是改写任务文本，让“同一段视频”在新任务下变成：

- 失败样本；
- 部分成功样本；
- near-miss 样本。

于是一个原本只能提供 “5/5 成功” 标签的视频，被扩展成了多种 reward 水平的训练样本。  
论文把这个过程类比为受 HER 启发的反事实数据构造，但这里重标注的是 **任务文本** 和 **目标分数**，不是目标状态本身。

### 7.2 形式化理解

设原始视频为 `v`，原始成功任务为 `t^+`，其标签为最高分：

```math
(v, t^+, 5)
```

经过反事实重标注，可以生成新的训练样本：

```math
(v, \tilde t_j, \tilde y_j)
```

其中：

- `\tilde t_j` 是重新提出的任务文本；
- `\tilde y_j` 是该视频在该任务下对应的离散进度分数；
- `\tilde y_j` 可能是 1、2、3、4，而不是 5。

### 7.3 具体流程

根据论文描述，重标注流程大致包括：

1. **Prompt rewriting**：先把原始任务描述的拼写和语法规范化，但不改变语义；  
2. **Negative example generation**：为成功视频生成不匹配的任务或部分匹配的任务；  
3. **Verification**：再用 VLM/LLM 验证新任务与视频的匹配程度以及分数是否合理；  
4. **Invariant text perturbation**：为同一任务再生成多个语义不变的 paraphrase，提高文本鲁棒性。

### 7.4 这一步为什么重要

如果没有这一步，reward model 训练将面临一个严重问题：

- 看到的大多数视频都是成功；
- 模型很容易学成“只要像机器人操作视频就给高分”；
- 但它学不会“同样的视频，对另一个任务文本应该低分”。

counterfactual relabeling 实际上在逼模型学一个真正的条件奖励函数：

```math
r = f(v, t)
```

而不是只看视频内容、不看任务条件的打分器。

---

## 8. 训练细节

作者在整理好的语料上微调 Qwen2.5-VL，分别训练 3B 和 7B 模型去预测 5 档进度标签。  
训练配置包括：

- 冻结 vision backbone；
- 微调 fusion 和 LLM 层；
- 学习率为 `3e-6`；
- weight decay 为 `0.05`；
- 有效 batch size 为 `64`（通过 gradient accumulation 实现）。

这一选择很合理，因为 reward model 更需要：

- 对 rollout 全局语义和任务条件建立对齐；
- 而不是彻底重学视觉表征。

---

## 9. benchmark 到底测什么

RoboRewardBench 不是简单测分类准确率。它要回答的是：

> 一个 VLM 作为机器人 reward model，能否在完整 rollout 上给出合理进度评分？

作者比较了 20 个 open-weight 和 proprietary VLM，并按 **mean win rate** 排名。  
直觉上，mean win rate 可以理解为：

> 随机与另一个模型 head-to-head 比较时，该模型更接近真实 reward 的概率。

按论文结果，RoboReward VLM 7B 和 3B 分列前两名，mean win rate 分别为 0.881 和 0.758，后面才是 Gemini 2.5 Flash、Qwen2.5-VL 72B、Gemini 2.5 Pro 等更大模型。

这说明一个重要事实：

> **机器人 reward 建模不是“模型越大越行”，而是高度依赖训练分布与任务形式。**

也就是说，针对机器人 reward 场景进行专门训练，比直接拿更大通用 VLM 零样本打分更有效。

---

## 10. 为什么这篇论文很重视“离线 reward benchmark”

很多人会问：reward model 离线指标高，为什么就一定对 RL 有用？

这篇论文正面回答了这个问题。  
作者先在 Robomimic 上做了 controlled experiment，结果发现：

- reward model 的 MAE 越低；
- 用它做下游 RL 时，最终 policy 表现越好；

并且二者相关系数约为 `r = 0.83`。

这意味着，RoboRewardBench 的意义不只是“模型比较”，而是：

> 它在相当程度上预测了这个 reward model 用于真实 RL 时会不会有效。

这也是论文最有说服力的一点。

---

## 11. 真实机器人 RL 结果说明了什么

作者把 RoboReward 3B 用在真实机器人 RL 中，和以下三种设置比较：

1. **Base diffusion policy before RL**  
2. **DSRL + Oracle human rewards**  
3. **DSRL + RoboReward VLM 3B**  
4. **DSRL + Qwen2.5-VL Instruct 3B（零样本）**

在两个真实任务上，结果是：

- `pick-and-place mushroom`：  
  - base policy：20%  
  - human reward：75%  
  - RoboReward 3B：45%  
  - base Qwen 3B reward：5%

- `open drawer`：  
  - base policy：60%  
  - human reward：80%  
  - RoboReward 3B：70%  
  - base Qwen 3B reward：10%

这个结果非常关键，说明三件事：

### 第一，差的 reward model 会直接伤害 RL
base Qwen 3B 作为零样本 reward，不但没有帮助，反而比不做 RL 更差。

### 第二，一个专门训练过的 reward model 即使没有达到人类水平，也能明显改善 policy
RoboReward 3B 仍然低于 human reward，但已经显著优于 base policy 和 base VLM reward。

### 第三，reward model 的“机器人适用性”比通用视觉语言能力更重要
同样是 VLM，专门为机器人 reward 训练过和没训练过，差距非常大。

---

## 12. 这篇论文的方法论价值

我认为这篇论文最值得记住的不是某一个数字，而是它建立了一个比较完整的范式。

### 12.1 先回答“什么 reward 更适合被学出来”
作者先在仿真分析了 binary / continuous / discrete progress 三种 reward 形式，而不是一上来就直接训练模型。  
这个设计很扎实，因为它先决定了“该学什么”。

### 12.2 再回答“数据分布不对怎么办”
通过 counterfactual relabeling，把 success-heavy 数据转成适合 reward learning 的数据。

### 12.3 最后回答“离线 reward 准确度到底有没有用”
通过真实 RL 实验，证明更好的 reward model 会带来更强的 policy improvement。

这条链是完整的：

```math
\text{reward formulation choice}
\rightarrow
\text{dataset construction}
\rightarrow
\text{reward model training}
\rightarrow
\text{offline benchmark}
\rightarrow
\text{real-robot RL}
```

---

## 13. 它和传统 reward learning / VLM-as-a-judge 的区别

这篇工作的特点，不只是“拿 VLM 来打分”，而是把 reward model 的要求定得更严格：

### 13.1 输入是整段 rollout，而不是单帧图像
也就是说，模型要看完整机器人过程，而不是只看结果帧。

### 13.2 输出是离散 progress，而不是偏好对比
它不是直接做 preference model，而是学习一个可用于 RL 的标量化 progress reward。

### 13.3 强调任务-视频条件一致性
同一个视频在不同任务文本下，reward 应该不同；counterfactual relabeling 正是在逼模型学这一点。

---

## 14. 这篇论文的局限

论文也有明显边界。

### 14.1 主要面向短时程 real-robot tasks
论文自己提到，他们的模型在 **short-horizon robotic tasks** 上表现突出。  
这意味着更长时序、需要复杂因果判断的任务，仍然是开放问题。

### 14.2 输出仍然是 episode-level progress
虽然比二值成功更细，但它仍然是整段 rollout 的一个离散分数，不是 step-wise dense reward。

### 14.3 依赖 LLM/VLM 管道做 counterfactual relabeling
这套流程本身也会引入标注噪声，需要靠 verification 环节校准。

---

## 15. 最终总结

如果只用一句话概括这篇论文：

> RoboReward 证明了，面向机器人 RL 的通用视觉-语言奖励模型不是直接拿现成大 VLM 零样本打分就能得到的；要想让 reward 真正可用，需要先选对 reward 形式（离散 progress）、再构造覆盖成功/失败/近似成功的数据分布、最后在真实机器人数据上专门训练和评测。

再压缩一点：

> 这篇论文把“VLM 能不能当机器人 reward model”这个问题，从一个模糊想法，推进成了一个**可训练、可评测、可用于真实 RL**的系统范式。
