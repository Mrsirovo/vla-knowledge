# LWD：部署即训练的机群级 VLA 后训练

> 论文：**Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies**  
> AGIBOT Finch / Shanghai Innovation Institute 等，2026  
> 算法背景：[`../rl/IQL&DIVL.md`](../rl/IQL&DIVL.md)（IQL → DIVL）  
> 相关对照：[`ALOE.md`](./ALOE.md)、[`pi0.6*.md`](./pi0.6*.md)、[`RL-Token.md`](./RL-Token.md)  
>
> 这份文档重点整理：
>
> 1. 为什么要把 **deployment 变成训练回路**，而不是终点  
> 2. 机群异构、稀疏奖励下如何做价值学习（DIVL）  
> 3. flow-based VLA 如何用 **QAM** 做策略提取  
> 4. offline → online 统一目标、与 imitation flywheel / Recap / ALOE 的差别

---

# 1. 一句话概括

LWD 的核心主张是：

> **部署不是训练的终点，而是数据飞轮的一环：把机群上的成功、失败、部分进度、恢复与人类干预，全部用同一套 offline-to-online RL 目标吃进一个通才 VLA，再 redeploy，持续改进。**

算法上拆成两块：

| 组件 | 作用 |
|------|------|
| **DIVL** | 从异构、稀疏、off-policy 的机群 replay 里学稳价值 |
| **QAM** | 把 critic 的动作梯度，稳定灌进 flow-matching 动作头 |

实验：16 台双臂机器人、8 个真实任务（含 3–5 分钟长程），**同一个** generalist policy 平均成功率约 **95%**，长程任务提升最大。

---

# 2. 问题：通才预训练不够，部署分布一直在变

预训练 VLA 给了广覆盖，但真实部署会不断遇到：

- 新物体 / 新布局 / 用户偏好  
- 长尾失败与恢复路径  
- 人类偶发干预与纠错  
- 多任务、多机异步、奖励稀疏且延迟

固定 offline demo 覆盖不了这些。常见两种不够用的做法：

## 2.1 只做 imitation flywheel

部署 → 筛“高质量成功轨迹” → BC 下一版策略。

问题：

- 失败、部分成功、干预片段被扔掉或弱用  
- 缺少从 **outcome** 反推“哪段动作更好”的机制  
- 长程任务上 compounding error 更痛

## 2.2 只做 specialist 在线 RL

单任务、短地平线、甚至从零训小策略，能涨点，但：

- 难保住通才能力  
- 难复用大规模 offline / 历史 off-policy 缓冲  
- 对整机群多任务同时后训练不友好

因此 LWD 要回答的是：

> **如何在机群部署分布上，持续后训练「一个」generalist flow-VLA，并稳定利用异构体验？**

---

# 3. 系统形态：Offline-to-Online 数据飞轮

```text
预训练 VLA
    │
    ▼
Offline RL 初始化（B_off：demo / 历史 rollout / 失败周边探索）
    │
    ▼
┌──────────────── 机群部署 ────────────────┐
│ 多机器人执行多任务                          │
│ 成功 / 失败 / 恢复 / 人类干预 → B_on       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        中心 learner：在 B_off ∪ B_on 上
        更新 V_ψ、Q_ϕ、flow policy f_θ
                   │
                   ▼
              推送新 checkpoint → 再部署
```

要点：

1. **Offline 与 Online 用同一套 RL 目标**（降低 offline critic 过保守、online 校准差的 mismatch）  
2. 学的是 **一个多任务 generalist**，不是每任务一个 specialist  
3. 数据不先滤成“示范子集”，而是用价值学习消费全谱体验

---

# 4. 算法一：DIVL（价值侧）

> 细节公式与 IQL 对照见 [`../rl/IQL&DIVL.md`](../rl/IQL&DIVL.md)。这里只写在 LWD 语境下“为什么需要它”。

## 4.1 机群 replay 让标量 critic 很难

同一 \((s,a)\) 在不同任务、不同干预密度、不同阶段下，回报可以：

- 多峰  
- 重尾  
- 成功稀少

标量 expectile / 均值容易把“罕见但可复现的成功模式”抹平。

## 4.2 DIVL 做法（直觉）

继承 IQL 的 **in-support 隐式改进**（不做 \(\max_a Q\) 的 OOD 最大化），但把 \(V(s)\) 升级成 **分布**：

```text
数据中的 Q(s, a) 直方图
  → 学 p_ψ(v | s)
  → 取 Quant_τ(V(s')) 做 TD bootstrap
  → 更新 Q_ϕ(s, a)
```

并可：

- 用价值分布熵调节 \(\tau(s)\)：不确定更保守  
- 长程离线用较大 \(n\)-step，加速稀疏成功回传；在线因干预混合常用更短 backup

在 LWD 里，DIVL 的职责是：

> **给异构机群数据一个不崩、还能保留高回报模态的 action-value 信号。**

---

# 5. 算法二：QAM（策略提取侧）

## 5.1 Flow VLA 为什么难直接做 RL

Flow / diffusion 动作头通过多步去噪生成 action chunk：

- 精确 likelihood 难算 → 标准 policy gradient / AWR 似然项不好写  
- 把 \(\nabla_a Q\) 反传到整条去噪链 → 贵且不稳

## 5.2 QAM 在做什么

目标仍是 KL 正则改进：

```math
\pi^*(a\mid s)
\propto
\pi_\beta(a\mid s)\,
\exp\big(Q_\phi(s,a)/\lambda\big)
```

但不对整条生成链做端到端 critic backprop，而是：

1. 在最终（去噪后）动作上取 \(\nabla_a Q\)  
2. 经 **adjoint** 变成沿参考 flow 轨迹的逐步回归目标  
3. 更新当前 flow 场 \(f_\theta\)，使其局部贴着“提高 \(Q\)、又不太离参考策略 \(\pi_\beta\)”的方向

直觉：

> **DIVL 给出“动作该往哪边更好”；QAM 把这个方向翻译成 flow matching 能吃的局部监督。**

这与 ALOE / Recap 的 advantage-weighted BC、RL Token 的小头锚定，同属“critic 指导 + 保守提取”，但 QAM 更直接吃 **连续动作空间的 critic 梯度**。

---

# 6. 训练阶段怎么配

## Stage 1：Offline RL pre-training

- Buffer：专家 demo、历史策略、失败周边 play / exploratory 数据  
- 同时训：分布价值 \(V_\psi\)、critic \(Q_\phi\)、flow policy  
- 目的：部署前有一个校准过的 critic + 可改进的初始策略

## Stage 2：Continuous online post-training

- 机群 rollout 写入 \(B_{on}\)（含可选人类干预）  
- Learner 混采 \(B_{off}\cup B_{on}\)  
- 周期性把新策略推回机器人

统一目标的意义：online 不是换一套算法“接着微调”，而是同一 DIVL+QAM 在部署分布上继续转。

---

# 7. 实验里应抓住的结论

设定：

- **16** 台 Agibot G1 双臂  
- **8** 个真实任务（功夫茶、鸡尾酒、果汁、装箱/鞋、grocery restock 等）  
- 多个 **3–5 分钟** 长程任务 + 需语义泛化的补货类短任务  
- **单一** generalist，而不是每任务专科

结果形态：

- 随机群在线经验累积，成功率持续上升，平均约 **95%**  
- **长程任务**相对 imitation / 其它后训练基线优势最大（信用分配与 stitching 更吃 RL）  
- 长程任务 **cycle time** 也可下降 → 不只更会做，还更快  
- 通常只需数小时量级真实交互即可看到实质改进（相对纯离线循环更贴部署分布）

---

# 8. 和其它 VLA-RL 工作怎么对照

| 方法 | 粒度 / 场景 | 价值 | 策略更新 | 通才？ |
|------|-------------|------|----------|--------|
| [Recap / π\*0.6](./pi0.6*.md) | 单任务大规模经验 | distributional V + advantage | advantage 条件 BC/flow | 偏 specialist |
| [π0.7](./pi0.7.md) | 通才蒸馏 | 用 metadata 条件化混合数据 | 条件生成 | 是（蒸馏 RL 经验） |
| [RL Token](./RL-Token.md) | 关键阶段、小时级 | 小 critic | 小 actor 锚定 VLA | 冻大模 + 局部头 |
| [ALOE](./ALOE.md) | HITL 真实后训练 | chunk TD + 悲观 ensemble | advantage-weighted flow | 任务级后训练 |
| **LWD** | **机群部署飞轮** | **DIVL** | **QAM** | **多任务 generalist** |

和 ALOE 的相近点：

- 都承认真实数据是异策略、碎片化、含干预  
- 都坚持要学 **action-level** 价值，而不是只做 trajectory progress

差别：

- ALOE 强调 Q-chunking + 悲观 ensemble + AWR 式提取  
- LWD 强调 **机群飞轮 + DIVL 分布价值 + QAM 对 flow 的梯度式提取**，并明确训 **跨任务单一通才**

和 imitation flywheel 的一句话差别：

> LWD 用 RL 消费“全谱部署体验”；imitation flywheel 主要消费“筛过的好动作标签”。

---

# 9. 实践理解：部署式 RL 的最小心智模型

```text
机群产生异构经验
  → DIVL：在 support 内学稳、可多峰的 Q
  → QAM：把 ∇Q 变成 flow 的局部回归
  → 同一目标贯穿 offline 与 online
  → redeploy 继续采数
```

若只记三点：

1. **系统**：deployment = continual training data source（机群共享策略）  
2. **价值**：DIVL = IQL 精神 + 分布/分位数，扛异构稀疏回报  
3. **策略**：QAM = 不把 flow 当可微黑盒硬反传，而用 adjoint 局部匹配

---

# 10. 参考文献与资源

- Wang et al., *Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies*, arXiv:2605.00416, 2026.  
- [项目页 / 博客](https://finch.agibot.com/research/lwd)  
- [arXiv HTML](https://arxiv.org/html/2605.00416)  
- DIVL 展开：[`../rl/IQL&DIVL.md`](../rl/IQL&DIVL.md)  
- QAM 原始思路：Li & Levine 等关于 flow policy 的 adjoint matching / Q-learning 工作（文中引用为 QAM）
