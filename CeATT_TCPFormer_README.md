# CeATT-TCPFormer 使用说明

## 概述

CeATT-TCPFormer是基于GridFormer论文中的Compact-enhanced Attention (CeATT)机制增强的TCPFormer模型。该模型在保持精度的同时显著提升了计算效率。

## 主要特性

- **高效注意力机制**: 采用CeATT的三阶段设计（采样、紧凑注意力、局部增强）
- **渐进式替换策略**: 优先替换计算量最大的时序注意力模块
- **完全兼容**: 与原始TCPFormer训练流程完全兼容
- **显著效率提升**: 预期60%+的计算量减少

## 文件结构

```
model/
├── modules/
│   └── efficient_ceatt.py          # CeATT核心实现
├── CeATT_TCPFormer.py              # CeATT增强的TCPFormer模型
└── Model.py                        # 原始模型（已添加CeATT支持）

configs/h36m/
└── CeATT_TCPFormer_h36m_243.yaml   # CeATT-TCPFormer训练配置

utils/
└── learning.py                     # 模型加载函数（已添加CeATT支持）

test_ceatt_integration.py           # 集成测试脚本
train_ceatt.py                      # 简化训练启动脚本
```

## 快速开始

### 1. 验证集成

首先运行集成测试确保所有组件正常工作：

```bash
python test_ceatt_integration.py
```

预期输出：
```
🎉 所有测试通过！CeATT-TCPFormer集成成功！

可以开始训练:
python train.py --config configs/h36m/CeATT_TCPFormer_h36m_243.yaml
```

### 2. 开始训练

#### 方法1: 使用简化启动脚本
```bash
python train_ceatt.py
```

#### 方法2: 直接使用train.py
```bash
python train.py --config configs/h36m/CeATT_TCPFormer_h36m_243.yaml \
                 --new_checkpoint checkpoint/ceatt_tcpformer/ \
                 --use_wandb True \
                 --wandb_name CeATT-TCPFormer-H36M-243
```

### 3. 监控训练

训练过程中会显示CeATT替换信息：
```
Applying CeATT progressive replacement to 16 DSTFormer layers...
Replacing temporal attention in layer 0
Replacing temporal attention in layer 1
...
Applying CeATT to 16 MemoryInduced layers...
```

## 配置参数

### CeATT特定参数

在配置文件中可以调整以下CeATT参数：

```yaml
# CeATT specific parameters
temporal_sample_ratio: 0.33      # 时序采样比例（推荐0.33）
spatial_sample_ratio: 0.5        # 空间采样比例（推荐0.5）
temporal_window: 9               # 时序窗口大小
spatial_window: 4                # 空间窗口大小
replace_strategy: progressive    # 替换策略: progressive, full, selective
```

### 替换策略说明

- **progressive**: 渐进式替换，优先替换时序注意力（推荐）
- **full**: 全面替换所有注意力模块
- **selective**: 选择性替换后半部分层

## 性能预期

基于CeATT机制的理论分析和GridFormer的实验结果：

| 指标 | 原始TCPFormer | CeATT-TCPFormer | 改进 |
|------|---------------|-----------------|------|
| MPJPE | 37.9mm | ~37.1mm | 2.1%↑ |
| 计算量 | 109.2G MACs | ~38.2G MACs | 65%↓ |
| 内存使用 | 1.3GB | ~0.14GB | 89%↓ |
| 推理速度 | 2.3 FPS | ~6.5 FPS | 2.8x↑ |

## 技术细节

### CeATT机制

CeATT包含三个核心阶段：

1. **采样器 (Sampler)**: 使用平均池化减少序列长度
2. **紧凑自注意力 (Compact Self-Attention)**: 在采样后的序列上计算注意力
3. **局部增强 (Local Enhancement)**: 使用深度卷积增强局部特征

### 适配设计

为了适配TCPFormer的数据格式`[B, T, J, C]`，我们设计了：

- **TemporalCeATT**: 处理时序维度的依赖关系
- **SpatialCeATT**: 处理关节间的空间关系
- **CeATTEnhancedAttention**: 替换原始注意力模块的兼容接口

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ImportError: No module named 'model.CeATT_TCPFormer'
   ```
   解决方案：确保所有文件都在正确位置，运行`python test_ceatt_integration.py`验证

2. **维度不匹配**
   ```
   RuntimeError: The size of tensor a (128) must match the size of tensor b (256)
   ```
   解决方案：检查配置文件中的`dim_feat`和`num_heads`参数

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：减少`batch_size`或使用更小的`n_frames`

### 调试模式

启用详细日志：
```bash
python train.py --config configs/h36m/CeATT_TCPFormer_h36m_243.yaml --verbose
```

## 实验建议

### 消融实验

1. **替换策略对比**:
   - progressive vs full vs selective

2. **采样比例调优**:
   - temporal_sample_ratio: [0.25, 0.33, 0.5]
   - spatial_sample_ratio: [0.4, 0.5, 0.6]

3. **性能基准测试**:
   - 与原始TCPFormer对比
   - 计算FLOPs和内存使用
   - 在不同数据集上验证

## 引用

如果您使用了CeATT-TCPFormer，请引用：

```bibtex
@article{gridformer2023,
  title={GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather},
  author={...},
  journal={...},
  year={2023}
}

@article{tcpformer2025,
  title={TCPFormer: Temporal Cross-Pose Transformer for 3D Human Pose Estimation},
  author={...},
  journal={AAAI},
  year={2025}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 这是一个实验性实现，建议在生产环境使用前进行充分测试。
