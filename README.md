# Vision-Language-Action (VLA) Knowledge Base

本仓库系统整理 **VLA 的发展脉络、代表性模型与关键技术**，帮助研究者快速入门与深入。内容覆盖机器人操作（manipulation）任务，涵盖近期重要综述与开源项目。

- 📜 历史纵览：`history/timeline.md`
- 🧠 主要模型：`models/`
- 🛠️ 关键技术（含 LoRA、[MEM](techniques/MEM.md)）：`techniques/`
- 🎯 VLA + 强化学习：`vla-rl/`
- 🤝 参与贡献：`CONTRIBUTING.md`

> 主线参考综述：
> - Shao et al., **Large VLM-based VLA Models for Robotic Manipulation: A Survey**, 2025  
> - Zhong et al., **A Survey on Vision-Language-Action Models: An Action Tokenization Perspective**, 2025  
> - Ma et al., **A Survey on VLAs for Embodied AI**, 2024  

---

## 快速导航

- 入门模型：RT-1、RT-2、RT-X、Octo、OpenVLA、π₀ / π₀.₅  
- 关键技术：LoRA/PEFT、语言-行动规划（SayCan/Code-as-Policies）、[MEM 长短期记忆](techniques/MEM.md)、动作标记化、加速与部署（VLA-Cache、SP-VLA、EfficientVLA）  
- VLA-RL：π\*0.6 / Recap、[π0.7](vla-rl/pi0.7.md)、[RL Token](vla-rl/RL-Token.md)、[ALOE](vla-rl/ALOE.md)、[LWD](vla-rl/LWD.md)、GR-RL、TGRPO、GRAPE  
- 数据资源：Open X-Embodiment、DROID、CALVIN

---

## 使用方法

```bash
git clone git@github.com:Mrsirovo/vla-knowledge.git
cd vla-knowledge-base
```

---

## 许可证

- 文本与整理：CC-BY 4.0（引用时请保留来源）  
- 具体论文/代码以各自许可证为准
