# Vision-Language-Action 模型比较

> 本文档总结了近年主要 VLA 模型的核心特征、创新点与局限性，帮助理解其演进脉络与未来趋势。

---

## 总览表

| 模型 | 提出年份 | 参数规模 | 数据规模 | 输入 | 输出 | 核心创新 | 主要局限 |
|------|----------|----------|----------|------|------|----------|----------|
| **RT-1** | 2022 | ~300M | 130k 真实演示 | 图像 + 语言 | 离散动作 token | 首个大规模 Transformer 策略，验证 NLP 范式在机器人可行性 | 单一平台，离散动作精度不足 |
| **RT-2** | 2023 | ~1.3B | 机器人数据 + 大规模互联网图文 | 图像 + 语言 | 动作 token（“actions as language”） | 引入 VLM 语义知识，支持零样本语义任务 | 离散化误差，训练成本极高 |
| **RT-X (Open X-Embodiment)** | 2023 | 多版本 (数亿至十亿) | 22 种机器人，21 机构，百万级演示 | 图像 + 语言 | 动作 token / 控制指令 | 跨平台数据集标准化，首次实证跨体现正迁移 | 平台差异过大时迁移有限，训练开销大 |
| **Octo** | 2024 | ~1B | ~800k 演示 | 图像 + 语言 + 目标图像 | 动作 token | 开源通用策略，强调快速微调，支持图像目标条件 | 泛化能力有限，受限于数据覆盖度 |
| **OpenVLA** | 2024 | 7B | 970k 真实演示 | 图像 + 语言 | 动作 token（离散→连续解码） | 双视觉编码器 (DINOv2 + SigLIP)，开源，支持 LoRA | 离散化误差，未见平台需微调，训练资源需求大 |
| **OpenVLA-OFT** | 2025 | 基于 OpenVLA (7B) | 下游任务数据 + LIBERO | 图像 + 语言 + 可扩展传感 | 动作 chunk（连续值） | 优化微调方案：并行解码、动作 chunk、连续表示、L1 回归 | 平台差异仍需适配，延迟挑战仍在 |
| **π₀** | 2024 | ~7B | 大规模跨平台演示 | 图像 + 语言 | 连续动作（flow matching + chunking） | 首次将 flow matching 应用于 VLA，支持高频连续控制 | 训练开销大，泛化仍有限 |
| **π₀.₅** | 2025 | ~7B | 多源数据 (机器人 + 网络 + 子任务标注) | 图像 + 语言 | 子任务 token + 动作 chunk | 混合推理：先语义子任务，再动作流生成；强调开放世界泛化 | 语义预测错误会级联，计算开销大 |

---

## 模型演进脉络

1. **从多任务到语义泛化**  
   - RT-1 → RT-2：从大规模演示的模仿学习，到引入 VLM 知识实现语义泛化。  

2. **从单一平台到跨体现**  
   - RT-X：首次整合跨平台数据，证明共享数据可带来正迁移。  
   - Octo：强调快速适配，推动开源生态。  

3. **开源与效率化**  
   - OpenVLA：首个开源 7B VLA 模型，支持 PEFT 和量化，降低使用门槛。  
   - OpenVLA-OFT：进一步优化微调与推理，使其更实用。  

4. **连续控制与开放世界**  
   - π₀：解决动作 token 离散化精度不足的问题，引入 flow matching + chunking，实现高频连续控制。  
   - π₀.₅：在此基础上引入语义 + 控制混合推理，推动开放世界泛化。  

---

## 优缺点对比

- **RT 系列**  
  - 优点：率先验证 Transformer 策略在机器人操作的可行性，RT-2 开启语义迁移。  
  - 缺点：离散动作限制明显，闭源，训练成本高。  

- **RT-X & Octo**  
  - 优点：跨平台数据和开源尝试，推动社区协作。  
  - 缺点：标准化牺牲精度，泛化仍受限。  

- **OpenVLA & OFT**  
  - 优点：开源、社区可用，LoRA 等技术降低门槛，OFT 提高效率与适配性。  
  - 缺点：视觉域泛化问题未解决，仍需高资源训练。  

- **π 系列**  
  - 优点：方法论创新（flow matching、语义+动作混合推理），推动向开放世界发展。  
  - 缺点：开销大，语义子任务预测仍有误差瓶颈。  

---

## 总结趋势

- **范式转变**：从专用模仿学习 → 大模型 Transformer → VLM 知识迁移 → 基座模型开源化。  
- **泛化路径**：先解决多任务（RT-1），再解决语义泛化（RT-2），然后跨体现（RT-X）、开放生态（Octo）、再到连续控制与开放世界泛化（π 系列）。  
- **效率化需求**：OFT 等工作表明，未来重点是如何降低延迟、提升吞吐，同时保持泛化。  
- **未来方向**：  
  1. 融合网络知识与机器人演示的高效训练框架  
  2. 更鲁棒的跨体现迁移与视觉泛化  
  3. 实时可部署的连续控制方案  
  4. 更强的语义 + 控制联合推理  

---

## 参考文献
- Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, RSS 2023.  
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv 2023.  
- Open X-Embodiment Collaboration et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, arXiv 2023.  
- Octo Team, *Octo: An Open-Source Generalist Robot Policy*, arXiv 2024.  
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv 2024.  
- Kim et al., *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)*, arXiv 2025.  
- Ye et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*, arXiv 2024.  
- Ye et al., *π₀.₅: a Vision-Language-Action Model with Open-World Generalization*, arXiv 2025.  
