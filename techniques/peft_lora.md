# 参数高效微调（PEFT）与 LoRA

## 背景
大规模模型（7B–70B 参数）在机器人和多模态学习中逐渐普及，但其全参数微调存在严重问题：  
- **显存需求过高**：例如 OpenVLA-7B 全参微调需要 >100GB 显存。  
- **存储成本大**：每个任务都要保存一份完整模型。  
- **数据效率差**：少样本任务往往不需要修改全部参数。  

**参数高效微调（PEFT）** 的目标：只更新小部分参数或新增少量参数模块，即可在不牺牲性能的情况下完成任务定制。  
其中最具代表性的方法是 **LoRA (Low-Rank Adaptation)**。

---

## 方法

### （入门解释）
想象一个 1000 页的书（大模型），如果你只需要改一个章节（新任务），没必要重写整本书。  
LoRA 的做法是：  
- 保留原始书本（冻结权重）。  
- 在关键章节旁加上“便签纸”（低秩矩阵）。  
- 训练时只修改便签，而不是动整本书。  

这样修改既轻便，又能灵活地加载或替换，适合不同的任务。

---

### （技术细节）
- **基本原理**：  
  对于一个线性层权重矩阵 $W \in \mathbb{R}^{d \times k}$，LoRA 不直接更新 $W$，而是新增一个低秩分解：

  $$W' = W + \Delta W = W + A B$$

  其中：

  - $A \in \mathbb{R}^{d \times r}$  
  - $B \in \mathbb{R}^{r \times k}$  
  - $r \ll \min(d, k)$ （典型取值 4–64）

- **训练机制**：  
  - 冻结原始 $W$。  
  - 仅训练 $A, B$。  
  - 参数量减少到原来的 ~0.1%–1%。  

- **推理机制**：  
  - 在推理时直接用 $W + AB$。  
  - 不增加显著延迟。  

---

## 使用案例
- **OpenVLA (2025)**：官方推荐在 RTX 3090/4090 上使用 LoRA 进行下游任务适配。  
- **Octo (2024)**：通过 LoRA 微调，研究者能在数千条演示数据上实现少样本任务学习。  
- **多机器人适配**：不同硬件（如 UR5、Franka Panda、Stretch）可通过各自 LoRA 权重快速迁移，而无需重新训练全模型。  
- **跨任务复用**：可以为“开门”、“搬运”、“整理桌面”分别保存不同的 LoRA 权重，并在运行时动态切换。  

---

## 意义
- **大幅降低资源需求**：使得 7B 模型能够在单张 24GB 显存的 GPU 上进行任务定制。  
- **少样本泛化能力**：即便只有几千条演示，也能通过 LoRA 得到较好的下游性能。  
- **模块化与可移植性**：LoRA 权重文件通常只有几十 MB，便于共享和组合。  
- **推动机器人研究民主化**：使更多实验室能在有限算力下使用大规模 VLA 基座模型。  

---

## 代码示例（PyTorch）

```python
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(out_f, r))
        self.B = nn.Parameter(torch.randn(r, in_f) * 1e-4)

    def forward(self, x):
        return self.base(x) + (x @ self.B.t() @ self.A.t())
