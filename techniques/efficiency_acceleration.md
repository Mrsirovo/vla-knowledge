# 高效与加速

## VLA-Cache
- 重用相邻时间步视觉 token
- 避免重复计算，推理提速 1.5x+

## SP-VLA
- 时空感知调度：关键步骤用大模型，其他用小模型
- Token 剪枝减少冗余

## EfficientVLA
- 训练无关加速与压缩方法
- 包含 Mixture-of-Layers（MoLE-VLA）
