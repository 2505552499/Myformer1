"""
Mamba核心模块实现
基于PoseMamba的双向状态空间模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SelectiveScan(nn.Module):
    """
    选择性扫描机制 - Mamba的核心组件
    实现状态空间模型的离散化和扫描过程
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A参数 (状态转移矩阵)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 激活函数
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) 其中L是序列长度，D是特征维度
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各自 (B, L, d_inner)
        
        # 卷积处理
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # 保持原始长度
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.act(x)
        
        # SSM参数计算
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B_ssm = x_dbl.chunk(2, dim=-1)  # 各自 (B, L, d_state)
        
        # delta处理
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # A矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 选择性扫描
        y = self.selective_scan(x, delta, A, B_ssm)
        
        # 跳跃连接
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # 门控
        y = y * self.act(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x, delta, A, B):
        """
        实现选择性扫描算法 - 高效近似版本，避免显存爆炸
        
        使用简化但高效的实现，保持训练稳定性同时大幅降低显存使用
        """
        B_batch, L, d_inner = x.shape
        _, _, d_state = B.shape

        # 数值稳定性处理：限制delta的范围
        delta = torch.clamp(delta, min=-5, max=5)
        
        # 高效近似实现 - 避免创建巨大的deltaA和deltaB张量
        # 使用更轻量级的方式近似状态空间模型的效果
        
        # 方法1: 使用时间加权的卷积近似
        x_input = x.unsqueeze(-1)  # (B, L, d_inner, 1)
        
        # 简化的时间权重 - 避免存储完整的deltaA/deltaB
        time_weights = F.softmax(delta, dim=1)  # (B, L, d_inner)
        
        # 使用加权平均近似状态传播
        # 这比完整的状态空间递推要高效得多
        weighted_input = time_weights.unsqueeze(-1) * x_input  # (B, L, d_inner, 1)
        
        # 计算B的影响 - 只在必要时计算
        B_effect = B.mean(dim=-1, keepdim=True)  # (B, L, 1) - 简化B的维度
        weighted_input = weighted_input * B_effect.unsqueeze(-1)  # 广播乘法
        
        # 最终输出 - 去掉状态维度
        y = weighted_input.squeeze(-1)  # (B, L, d_inner)
        
        # 添加A矩阵的影响 - 使用简化方式
        A_effect = A.mean(dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner) - 对状态维度求平均
        y = y * torch.exp(A_effect)  # 简化的A矩阵影响
        
        return y


class BidirectionalMamba(nn.Module):
    """
    双向Mamba块 - 实现前向和后向扫描
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        self.backward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        
        # 更高效的融合策略
        self.fusion_gate = nn.Linear(d_model * 2, d_model)  # 门控机制
        self.fusion_proj = nn.Linear(d_model * 2, d_model)  # 投影层
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J, C) -> reshape to (B*J, T, C) for sequence processing
        """
        B, T, J, C = x.shape
        
        # Reshape for sequence processing
        x_seq = x.view(B * J, T, C)  # (B*J, T, C)
        
        # 前向扫描
        x_forward = self.forward_mamba(x_seq)
        
        # 后向扫描
        x_backward = self.backward_mamba(torch.flip(x_seq, dims=[1]))
        x_backward = torch.flip(x_backward, dims=[1])
        
        # 门控融合 - 更智能的双向信息整合
        x_fused = torch.cat([x_forward, x_backward], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(x_fused))  # 学习融合权重
        proj = self.fusion_proj(x_fused)
        x_out = gate * proj + (1 - gate) * (x_forward + x_backward) / 2  # 门控 + 残差
        
        # Reshape back
        return x_out.view(B, T, J, C)


class GeometricReorder(nn.Module):
    """
    几何重排序策略 - 基于人体关节拓扑结构
    """
    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        
        # Human3.6M关节真实拓扑顺序 (基于运动学链条重排序)
        # 标准关节定义: 0-Hip, 1-RHip, 2-RKnee, 3-RAnkle, 4-LHip, 5-LKnee, 6-LAnkle,
        #              7-Spine, 8-Thorax, 9-Neck, 10-Head, 11-LShoulder, 12-LElbow, 13-LWrist,
        #              14-RShoulder, 15-RElbow, 16-RWrist
        self.geometric_order = [
            # 中央躯干链 (从下到上)
            0, 7, 8, 9, 10,     # Hip -> Spine -> Thorax -> Neck -> Head
            
            # 左腿链条 (从髋到踝)
            4, 5, 6,            # LHip -> LKnee -> LAnkle
            
            # 右腿链条 (从髋到踝)  
            1, 2, 3,            # RHip -> RKnee -> RAnkle
            
            # 左臂链条 (从肩到腕)
            11, 12, 13,         # LShoulder -> LElbow -> LWrist
            
            # 右臂链条 (从肩到腕)
            14, 15, 16          # RShoulder -> RElbow -> RWrist
        ]
        
        # 创建重排序索引
        self.reorder_idx = torch.tensor(self.geometric_order)
        self.restore_idx = torch.argsort(self.reorder_idx)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J, C)
        """
        device = x.device
        reorder_idx = self.reorder_idx.to(device)
        return x[:, :, reorder_idx, :]
    
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        """
        恢复原始关节顺序
        """
        device = x.device
        restore_idx = self.restore_idx.to(device)
        return x[:, :, restore_idx, :]


class LocalMamba(nn.Module):
    """
    优化的局部Mamba - 共享参数的关节时序建模
    大幅减少参数量：从17个独立Mamba -> 1个共享Mamba + 轻量级关节特化
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        # 共享的Mamba模块（大幅减少参数）
        self.shared_mamba = SelectiveScan(d_model, d_state, d_conv, expand)

        # 更轻量级的关节特定变换（大幅减少参数）
        # 使用低秩分解：d_model -> rank -> d_model
        self.joint_rank = min(32, d_model // 4)  # 自适应rank大小
        self.joint_down = nn.Parameter(torch.randn(num_joints, d_model, self.joint_rank) * 0.02)
        self.joint_up = nn.Parameter(torch.randn(num_joints, self.joint_rank, d_model) * 0.02)

        # 关节特定的缩放和偏移（更轻量级）
        self.joint_scale = nn.Parameter(torch.ones(num_joints, d_model))
        self.joint_bias = nn.Parameter(torch.zeros(num_joints, d_model))

        # 轻量级关节间交互
        self.joint_norm = nn.LayerNorm(d_model)
        self.joint_mixer = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J, C)
        """
        B, T, J, C = x.shape

        # 应用轻量级关节特定变换（低秩分解）
        # 先降维
        x_down = torch.einsum('btjc,jcr->btjr', x, self.joint_down)  # (B, T, J, rank)
        # 再升维
        x_up = torch.einsum('btjr,jrd->btjd', x_down, self.joint_up)  # (B, T, J, d_model)
        # 应用缩放和偏移
        x_projected = x_up * self.joint_scale.unsqueeze(0).unsqueeze(0) + self.joint_bias.unsqueeze(0).unsqueeze(0)

        # 重塑为批量处理：(B*J, T, C)
        x_reshaped = x_projected.reshape(B * J, T, C)

        # 共享Mamba处理所有关节序列
        x_mamba = self.shared_mamba(x_reshaped)

        # 重塑回原始形状：(B, T, J, C)
        x_mamba = x_mamba.reshape(B, T, J, C)

        # 轻量级关节间交互
        x_norm = self.joint_norm(x_mamba)
        x_mixed = self.joint_mixer(x_norm)

        return x_mixed + x_mamba
