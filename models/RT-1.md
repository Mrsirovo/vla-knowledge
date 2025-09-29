# RT-1 (Robotics Transformer 1)

## 概览
- **全称**：RT-1: Robotics Transformer for Real-World Control at Scale  
- **提出时间**：2022 年 / arXiv 发布，之后作为 RSS 2023 会议论文公开 :contentReference[oaicite:0]{index=0}  
- **目标**：验证“Transformer + 大规模真实机器人数据”能否在多任务操控中获得较好的泛化与鲁棒性  
- **核心创新**：将视觉、语言和动作都 token 化，并在大规模多任务机器人演示上训练 Transformer 模型，使其具备零样本泛化与吸收异构数据的能力 :contentReference[oaicite:1]{index=1}  
- **输入 / 输出**：输入为历史图像帧 + 自然语言指令；输出为离散动作 token（控制机械臂 + 移动基座 + 终止信号） :contentReference[oaicite:2]{index=2}  
- **性能展示**：在 700 多条训练指令上达 ~97% 成功率；在未见组合任务 / 新环境 / 干扰环境上也表现出较好泛化能力 :contentReference[oaicite:3]{index=3}  

---

## 背景与动机
在 RT-1 之前，机器人操作模型大多依赖于针对单任务或少量任务的数据，且策略往往针对具体硬件或环境优化，难以扩展或泛化。  
NLP / 视觉领域成功经验表明：**大模型 + 多样化数据** 是泛化能力的关键，这个思路在机器人控制中尚未被充分验证。RT-1 的提出即是为了探索这一范式在机器人操作中的可行性。 :contentReference[oaicite:4]{index=4}  

此外，机器人真实数据难以采集、成本高昂，如何设计一个能“吸收”多源、异构数据的模型，是 RT-1 致力解决的问题。 :contentReference[oaicite:5]{index=5}  

---

## 方法

### 入门理解
RT-1 的基本直觉是：把语言、视觉、动作都变成一串“token”，然后用 Transformer 来学习这些 token 之间的映射关系。这样，模型能够从大规模多样化的机器人演示中学习不同任务之间共通性，从而在新任务、不同环境中具备泛化能力。

### 技术细节

| 模块 / 机制 | 作用 | 细节说明 |
|---|---|---|
| **视觉 + 语言 token 化** | 将图像 / 指令转为 token 便于 Transformer 处理 | 图像通过 EfficientNet + FiLM 层得到特征 → 展平为 token，语言 embedding 融入视觉 token 化过程 :contentReference[oaicite:6]{index=6} |
| **动作 token 化** | 将连续动作离散化，统一为 token 输出 | 机械臂运动 (7 维：x, y, z, 旋转 roll/pitch/yaw, gripper) + 移动基座 (3 维) + 模式开关维度，共离散化为 256 个 bin；并且使用一个额外 token 用于表示“控制 arm / base / 终止”模式 :contentReference[oaicite:7]{index=7} |
| **Token Learner（token 压缩）** | 减少输入 token 数量，加快推理 | 从视觉 token 中选择软组合以压缩为较少 token 供 Transformer 处理，提升效率 :contentReference[oaicite:8]{index=8} |
| **Transformer 模型** | 融合 token 进行语言-视觉-动作映射 | 使用标准 Transformer 架构（decoder 式 / 自回归结合因果遮蔽）进行 token‐to‐token 映射 :contentReference[oaicite:9]{index=9} |
| **训练目标与优化** | 学习 token 映射 | 使用交叉熵 (cross-entropy) 监督训练预测动作 token，最小化预测与真实 token 的差距 :contentReference[oaicite:10]{index=10} |
| **异构数据融合** | 吸收更多样本，提升泛化 | 在训练中加入仿真数据或其他机器人数据（如 Kuka bin-picking 数据）来增强泛化能力，且不会显著损害原任务性能 :contentReference[oaicite:11]{index=11} |

**流程**：
1. 输入一段历史 RGB 帧 + 语言指令  
2. 将图像与指令 token 化  
3. Transformer 输入这些 token，预测下一个动作 token  
4. 输出控制机械臂 / 移动基座 / 终止信号  
5. 在控制 loop 中闭环执行直到任务结束或终止 token  

---

## 使用案例与能力展示

- **见任务成功**：RT-1 在训练任务 (700 多条) 上达 ~97% 成功率 :contentReference[oaicite:12]{index=12}  
- **未见组合任务泛化**：对新组合的指令执行成功率 ~76% :contentReference[oaicite:13]{index=13}  
- **鲁棒性测试**：在含干扰物 (distractors) 和背景变化 (new scenes) 的条件下仍能保持较好成功率（83% 对抗干扰，59% 对抗背景变化）:contentReference[oaicite:14]{index=14}  
- **长时序任务**：在 SayCan 框架下执行长指令（最高 50 步）任务，在 Kitchen 场景中成功率 ~67% :contentReference[oaicite:15]{index=15}  
- **异构数据融合实验**：加入仿真 / 不同机器人数据 (如 Kuka bin-picking)，能提升泛化而不显著牺牲原始任务性能 :contentReference[oaicite:16]{index=16}  

---

## 局限性

- **仅依赖仿效学习 (imitation learning)**：受演示质量限制，无法超越示范者性能 :contentReference[oaicite:17]{index=17}  
- **离散动作表达限制**：离散 token 化可能在高精度连续控制任务中表现欠佳 :contentReference[oaicite:18]{index=18}  
- **组合能力受限**：在完全未见的动作模式或全新操作类型上，泛化能力仍有限 :contentReference[oaicite:19]{index=19}  
- **实时控制挑战**：虽然做了一些优化 (token compression 等)，Transformer 推理在控制环路中仍有延迟压力 :contentReference[oaicite:20]{index=20}  
- **数据 / 硬件依赖**：数据与机器人硬件类型集中 (Everyday Robots 平台)，跨平台泛化仍需进一步验证 :contentReference[oaicite:21]{index=21}  

---

## 意义与影响

- **范式验证**：RT-1 是第一个在 **真实机器人多任务操作 + Transformer 架构** 上取得成功泛化的工作。  
- **标杆模型**：它展示了“吸收海量异构数据 + token 化策略”在操作泛化上的潜力，为后续 RT-2、RT-X、OpenVLA 等提供了方法基础。  
- **工程价值**：通过 token 压缩 (Token Learner)、动作 token 化等设计，使得在有限硬件上可行执行。  
- **扩展潜力**：其异构数据吸收能力（仿真、不同机器人）为多平台共享策略提供示范；同时其设计也为未来将语义知识（语言-视觉模型）融入控制提供思路。  

---

## 参考文献
- Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, arXiv 2022 / RSS 2023. :contentReference[oaicite:22]{index=22}  
- RT-1 项目主页 (Robotics Transformer) :contentReference[oaicite:23]{index=23}  
