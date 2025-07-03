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
        实现选择性扫描算法
        """
        B_batch, L, d_inner = x.shape
        _, _, d_state = B.shape
        
        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # 初始化状态
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        # 扫描过程
        for i in range(L):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            y = h.sum(dim=-1)  # (B, d_inner)
            ys.append(y)
        
        return torch.stack(ys, dim=1)  # (B, L, d_inner)


class BidirectionalMamba(nn.Module):
    """
    双向Mamba块 - 实现前向和后向扫描
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        self.backward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        
        # 融合层
        self.fusion = nn.Linear(d_model * 2, d_model)
        
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
        
        # 融合
        x_fused = torch.cat([x_forward, x_backward], dim=-1)
        x_out = self.fusion(x_fused)
        
        # Reshape back
        return x_out.view(B, T, J, C)


class GeometricReorder(nn.Module):
    """
    几何重排序策略 - 基于人体关节拓扑结构
    """
    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        
        # Human3.6M关节拓扑顺序 (基于身体结构的重排序)
        # 0: 根节点(骨盆), 1-6: 左腿, 7-12: 右腿, 13-16: 躯干和头部
        self.geometric_order = [
            0,   # 根节点
            1, 2, 3,     # 左髋、左膝、左踝
            4, 5, 6,     # 右髋、右膝、右踝  
            7,           # 躯干
            8, 9, 10,    # 颈部、头顶、鼻子
            11, 12,      # 左肩、左肘
            13,          # 左腕
            14, 15,      # 右肩、右肘
            16           # 右腕
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

        # 轻量级关节特定变换（仅17*d_model*d_model参数）
        self.joint_projections = nn.Parameter(torch.randn(num_joints, d_model, d_model) * 0.02)

        # 轻量级关节间交互
        self.joint_norm = nn.LayerNorm(d_model)
        self.joint_mixer = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J, C)
        """
        B, T, J, C = x.shape

        # 应用关节特定变换
        x_projected = torch.einsum('btjc,jcd->btjd', x, self.joint_projections)

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
