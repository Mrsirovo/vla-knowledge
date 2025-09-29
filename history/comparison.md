# Vision-Language-Action 模型比较（含资源信息）

> 本文档总结了主要 VLA 模型的核心特征、创新点、训练规模与资源需求，帮助理解其演进脉络与未来趋势。

---

## 总览表

| 模型 | 年份 | 参数规模 | 数据规模 | 训练资源 (GPU × 时间) | 输入 | 输出 | 核心创新 | 局限性 |
|------|------|----------|----------|-----------------------|------|------|----------|--------|
| **RT-1** | 2022 | 未公开 (≈ 数亿级) | ~130k 真实演示 | 未公开 | 图像 + 语言 | 离散动作 token | 首个大规模 Transformer 策略，验证 NLP 范式在机器人可行性 | 单一平台，离散动作精度不足 |
| **RT-2** | 2023 | 多版本 (最大至 55B) | 机器人演示 + 大规模 Web 图文 | 未公开 | 图像 + 语言 | 动作 token（“actions as language”） | 引入 VLM 语义知识，支持零样本语义任务 | 离散化误差，训练成本极高 |
| **RT-X (Open X-Embodiment)** | 2023 | 未固定（多版本 Transformer） | 百万级演示，22 种机器人，21 个机构 | 未公开 | 图像 + 语言 | 动作 token / 控制指令 | 跨平台数据集标准化，首次实证跨体现正迁移 | 平台差异过大时迁移有限，训练开销大 |
| **Octo** | 2024 | 未公开 (≈ 1B 级) | ~800k 演示 | 未公开 | 图像 + 语言 + 目标图像 | 动作 token | 开源通用策略，强调快速微调，支持图像目标条件 | 泛化受限，受数据覆盖度影响 |
| **OpenVLA** | 2024 | 7B | ~970k 真实演示 | 64 × A100 GPU，15 天 | 图像 + 语言 | 动作 token → 连续控制 | 双视觉编码器 (DINOv2 + SigLIP)，开源，支持 LoRA | 离散化误差，未见平台需微调，训练资源需求大 |
| **OpenVLA-OFT** | 2025 | 基于 OpenVLA (7B) | 下游任务数据 + LIBERO | 8 × A100/H100，50K–150K 步，1–2 天 | 图像 + 语言 + 可扩展传感 | 动作 chunk（连续值） | 优化微调方案：并行解码、chunk、连续表示、L1 回归 | 平台差异仍需适配，延迟挑战仍在 |
| **π₀** | 2024 | ≈7B (估计) | 大规模跨平台演示 | 未公开 | 图像 + 语言 | 连续动作 (flow matching + chunking) | 首次将 flow matching 应用于 VLA，高频连续控制 | 训练开销大，泛化仍有限 |
| **π₀.₅** | 2025 | ≈7B (估计) | 多源数据（机器人 + 网络 + 子任务标注） | 未公开 | 图像 + 语言 | 子任务 token + 动作 chunk | 混合推理（语义子任务 + 动作流），开放世界泛化 | 语义预测错误会级联，计算开销大 |

---

## 模型演进脉络

1. **多任务到语义泛化**  
   - RT-1 → RT-2：从多任务大规模模仿学习到引入 VLM 知识。  

2. **跨体现与数据共享**  
   - RT-X：首次跨平台数据集标准化，验证正迁移可行性。  
   - Octo：强调快速微调与开源生态。  

3. **开源与效率化**  
   - OpenVLA：首个开源 7B 级 VLA，支持 LoRA/量化。  
   - OpenVLA-OFT：提出系统化微调 recipe，提高成功率和吞吐。  

4. **连续控制与开放世界**  
   - π₀：引入 flow matching，解决离散动作精度不足。  
   - π₀.₅：加入语义子任务推理，迈向开放世界泛化。  

---

## 优缺点对比

- **RT 系列**  
  - 优点：率先验证大规模 Transformer 策略，RT-2 引入 VLM 知识迁移。  
  - 缺点：离散动作限制明显，训练资源未披露，闭源。  

- **RT-X & Octo**  
  - 优点：推动跨平台与开源数据集共享。  
  - 缺点：资源消耗大，跨平台适配精度受限。  

- **OpenVLA & OFT**  
  - 优点：开源、社区可用；支持高效微调与量化；OFT 显著提升吞吐和任务成功率。  
  - 缺点：视觉泛化问题，训练资源需求仍高。  

- **π 系列**  
  - 优点：方法论创新，提升连续控制和开放世界能力。  
  - 缺点：计算资源与语义推理仍是瓶颈。  

---

## 总结趋势

- **范式演进**：模仿学习 → 大模型 Transformer → VLM 迁移 → 开源基座 → 高效微调 → 连续控制 / 开放世界。  
- **资源权衡**：训练资源需求从数十万样本 + 数百 GPU 小时，扩展到百万样本 + 数十至上百 GPU 天。  
- **未来方向**：  
  1. 更高效的跨体现迁移机制  
  2. 优化视觉泛化，避免 catastrophic forgetting  
  3. 降低训练与微调开销  
  4. 强化语义 + 控制联合推理  

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
