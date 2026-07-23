# VLA × 强化学习笔记

本目录整理 **用 RL / 类 RL 信号改进 VLA** 的代表性工作，文风偏方法链路（问题 → 思想 → 公式 → 落地），方便和 `models/`、`rl/` 对照阅读。

## Physical Intelligence 主线（建议按此顺序）

| 文档 | 一句话 |
|------|--------|
| [../techniques/MEM.md](../techniques/MEM.md) | 多尺度记忆：短期视频编码 + 长期语言摘要（π0.6-MEM / π0.7 底座） |
| [pi0.6*.md](./pi0.6*.md) | Recap：distributional value + advantage 条件，做整模经验后训练 → π\*0.6 |
| [RL-Token.md](./RL-Token.md) | 抽出 RL token，冻结大 VLA，小 actor-critic 小时级精修“最后一毫米” |
| [pi0.7.md](./pi0.7.md) | 多模态 strategy conditioning，把 demo / 失败 / RL rollout 蒸馏进可操控通才模型 |

关系简述：

```text
Recap 产出高吞吐 specialist 经验
        ↓ metadata 条件化
     π0.7 通才开箱对齐 specialist
        ↓ 若仍有接触瓶颈
     RL Token 局部在线精修
```

## 其他方法

| 文档 | 侧重 |
|------|------|
| [ALOE.md](./ALOE.md) | HITL 真实后训练：action-level off-policy evaluation + Q-chunking |
| [LWD.md](./LWD.md) | 机群部署飞轮：DIVL 价值 + QAM 提取，通才 offline-to-online RL |
| [GR-RL.md](./GR-RL.md) | progress critic 过滤数据 + 噪声空间在线 RL |
| [TGRPO.md](./TGRPO.md) | 轨迹级 GRPO 类更新 |
| [GRAPE.md](./GRAPE.md) | preference / 偏好驱动的 VLA 改进 |
| [roboreward.md](./roboreward.md) | 机器人奖励模型相关 |

## 经典离线 RL 背景

更偏算法本身的笔记在 [`../rl/IQL&DIVL.md`](../rl/IQL&DIVL.md)（LWD 的 DIVL 在此展开）。
