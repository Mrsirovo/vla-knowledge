# VLA 发展时间线

> 本文档梳理 Vision-Language-Action (VLA) 模型在机器人操作领域的主要发展路径，从 2021 年的早期跨模态探索，到 2025–2026 年大规模开源基座、高效推理与 VLA-RL 后训练。每个节点附有研究动机、方法亮点、意义、局限性与配图建议，便于后续在项目或演示中扩展。

---

## **2021 — CLIP 与跨模态预训练的启发**
- **动机**：自然语言与视觉表示的统一为机器人带来启示：能否用“语言 + 图像”直接指导机器人行动？  
- **方法**：CLIP (Radford et al., 2021) 在 4 亿图文对上预训练，实现零样本分类。  
- **意义**：虽然 CLIP 并非为机器人设计，但其**跨模态对齐能力**成为 VLA 模型的重要启发。  
- **图示建议**：CLIP 架构图（双编码器 + 对比学习），箭头指向机器人操作场景。  
- **引用**：Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.  

---

## **2022 — SayCan**
- **动机**：单纯 LLM 虽能理解任务，但无法保证物理可行性。  
- **方法**：SayCan = LLM (语言推理) + 技能库 (affordance model)。  
  - LLM 提供高层任务分解。  
  - 技能库约束可执行性并输出低层控制。  
- **技术特点**：首次把语言推理与机器人低层控制连接在真实场景。  
- **意义**：开创“语言规划 + 行动执行”的范式。  
- **局限性**：依赖人工定义的技能库，扩展性受限。  
- **图示建议**：LLM → 任务分解 → 技能库 → 机器人执行的流程图。  
- **引用**：Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, arXiv 2022.  

---

## **2022 — Perceiver-Actor**
- **动机**：探索多模态感知与机器人控制的一体化。  
- **方法**：基于 Perceiver IO 架构，输入语言 + 图像，输出低层连续动作。  
- **意义**：证明“统一 Transformer”可直接建模 VLA。  
- **局限性**：数据规模有限，泛化能力不足。  
- **引用**：Shafiullah et al., *Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation*, CoRL 2022.  

---

## **2022 — RT-1**
- **方法**：Google Robotics 提出 Robotics Transformer (RT-1)，在 ~130k 演示上训练。  
- **亮点**：  
  - 输入 = 语言 + 图像，输出 = 动作 token 序列。  
  - 多任务数据训练显著提升泛化。  
- **意义**：奠定了“大规模数据 + Transformer”在机器人中的可行性。  
- **局限性**：限于单一机器人平台。  
- **图示建议**：Transformer 模型图，左侧输入“语言+图像”，右侧输出“动作序列”。  
- **引用**：Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, RSS 2023.  

---

## **2023 — RT-2**
- **动机**：让机器人具备“互联网级知识”。  
- **方法**：在 RT-1 基础上，融合 VLM 预训练（网页图文数据）。  
- **亮点**：  
  - 零样本泛化：可执行未见过的语义任务（如“搬运卡通角色物体”）。  
- **意义**：突破语义理解瓶颈，首次展现“知识迁移到行动”。  
- **局限性**：计算成本高。  
- **图示建议**：VLM 语料 → 融合机器人数据 → RT-2 策略模型。  
- **引用**：Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robots*, arXiv 2023.  

---

## **2023 — Open X-Embodiment / RT-X**
- **动机**：解决单一实验室数据不足问题。  
- **方法**：21 机构联合，22 种机器人，百万级演示数据。  
- **成果**：  
  - 训练出的 RT-X 展示了跨机器人迁移。  
- **意义**：建立了“多机器人数据共享”的范式。  
- **局限性**：数据标注和分布仍有差异。  
- **图示建议**：多机器人图标（机械臂、移动机器人）→ 汇聚到统一 Transformer。  
- **引用**：Open X-Embodiment Collaboration, *RT-X*, arXiv 2023.  

---

## **2024 — Octo**
- **动机**：推动开源化，让研究者可直接使用大规模策略模型。  
- **方法**：~80 万演示，语言指令 + 目标图像输入。  
- **亮点**：开源通用策略，可快速在消费级 GPU 上微调。  
- **意义**：学术界首次拥有“可自由适配”的大型 VLA 模型。  
- **局限性**：数据质量仍受限于合作方。  
- **引用**：Octo Team, *Octo: An Open-Source Generalist Robot Policy*, arXiv 2024.  

---

## **2024 — DROID 数据集**
- **方法**：Google Research 发布 DROID 数据集。  
- **规模**：76k 轨迹，350 小时，564 场景，86 个任务。  
- **意义**：强调 in-the-wild 数据的重要性，提高泛化。  
- **图示建议**：不同环境（厨房、客厅、户外）的小插图。  
- **引用**：Chi et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, arXiv 2024.  

---

## **2025 — OpenVLA**
- **方法**：开源 7B VLA，97 万真实演示。  
- **亮点**：支持参数高效微调 (LoRA)，消费级 GPU 可适配。  
- **意义**：标志着“开放基座模型”在机器人中的落地。  
- **局限性**：推理速度和部署仍是挑战。  
- **图示建议**：7B 模型图，旁边标出“LoRA 插件”。  
- **引用**：Kim et al., *OpenVLA*, arXiv 2025.  

---

## **2025 — 高效推理方法**
- **背景**：7B 级别模型在机器人上推理延迟过高。  
- **代表方法**：  
  - **VLA-Cache**：缓存相邻时间步的 token。  
  - **SP-VLA**：任务阶段动态调度大小模型。  
  - **EfficientVLA**：总结无训练压缩与 Mixture-of-Layers。  
- **意义**：推动 VLA 向实时部署迈进。  
- **引用**：Xu et al., *VLA-Cache*, 2025; Li et al., *SP-VLA*, 2025; Yang et al., *EfficientVLA*, 2025.  

---

## **2025–2026 — VLA 的强化学习后训练**
- **动机**：通才 VLA“会做”不等于“又快又稳”；示范数据在接触丰富阶段噪声大。  
- **代表路线**：  
  - **MEM**：多尺度具身记忆——短期稠密视频历史 + 长期可压缩语言摘要，支撑约 15 分钟级任务（π0.6-MEM / π0.7 底座）。  
  - **Recap / π\*0.6**：distributional value + advantage 条件提取，整模吸收 online/offline 经验。  
  - **RL Token (RLT)**：压缩 VLA 表征为 RL token，轻量 actor-critic 在小时级数据上精修关键阶段。  
  - **ALOE**：真实 HITL 数据上做 action-level off-policy evaluation（Q-chunking + 悲观 ensemble）。  
  - **LWD**：机群部署飞轮，DIVL + QAM，持续后训练多任务通才 VLA。  
  - **π0.7**：多模态 context（语言 / metadata / 子目标图像）消歧混合质量数据，把 RL specialist 经验蒸馏进可操控通才模型。  
- **意义**：形成“记忆底座 → 大规模经验改进 → 通才蒸馏 → 局部在线精修 / 机群部署飞轮”的多层栈。  
- **笔记**：`techniques/MEM.md`、`vla-rl/pi0.6*.md`、`vla-rl/RL-Token.md`、`vla-rl/pi0.7.md`、`vla-rl/ALOE.md`、`vla-rl/LWD.md`  

---

# 未来展望
- **长时序记忆**：MEM 已给出短视频 + 长语言的实用方案；更长地平线与主动遗忘/检索仍待扩展。  
- **4D 感知**：从视频/点云学习动态表征。  
- **多机器人协作**：多智能体 VLA 模型。  
- **能效与压缩**：量化、蒸馏与 PEFT 的结合。  
- **多层经验学习**：通才蒸馏、阶段级在线 RL 与高层推理 RL 如何统一。

---
