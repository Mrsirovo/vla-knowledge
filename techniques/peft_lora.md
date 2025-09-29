# 参数高效微调（PEFT）与 LoRA

## 背景
- 大模型全参数微调成本高
- LoRA (Low-Rank Adaptation) 是一种高效 PEFT 技术

## 方法
- 冻结原始权重 \(W\)
- 新增低秩矩阵 \(\Delta W = A B\)
- 训练时仅更新 \(A, B\)，参数量大幅减少

## 优点
- 显存需求大幅下降
- 适合少样本任务
- 可模块化加载不同 LoRA 权重

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
```

## 影响
- 大多数任务性能不降
- 少样本场景下泛化更好
- 若任务差异极大或 rank 太小时性能可能下降

