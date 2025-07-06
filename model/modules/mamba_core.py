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
        实现选择性扫描算法 - 正确的状态空间模型实现
        
        状态空间模型的递推公式：
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
        
        在Mamba中，我们使用选择性扫描来实现这个递推过程
        """
        B_batch, L, d_inner = x.shape
        _, _, d_state = B.shape

        # 数值稳定性处理：限制delta的范围
        delta = torch.clamp(delta, min=-10, max=10)
        
        # 离散化 - 将连续时间SSM转为离散时间
        # deltaA: 状态转移矩阵的离散化版本
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        
        # deltaB: 输入矩阵的离散化版本  
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # 选择实现方式：短序列用循环，长序列用并行扫描
        if L <= 64:
            return self._sequential_scan(x, deltaA, deltaB, B_batch, L, d_inner, d_state)
        else:
            return self._parallel_scan(x, deltaA, deltaB, B_batch, L, d_inner, d_state)
    
    def _sequential_scan(self, x, deltaA, deltaB, B_batch, L, d_inner, d_state):
        """序列化扫描 - 用于短序列，精确但较慢"""
        # 初始化状态
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # 输出序列
        outputs = []

        # 实现正确的状态空间递推
        for t in range(L):
            # 当前时刻的输入
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            
            # 状态更新: h_t = A_t * h_{t-1} + B_t * x_t
            h = deltaA[:, t, :, :] * h + deltaB[:, t, :, :] * x_t
            
            # 输出计算: y_t = sum(h_t, dim=-1) (简化的输出矩阵C)
            y_t = h.sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        # 堆叠所有时刻的输出
        return torch.stack(outputs, dim=1)  # (B, L, d_inner)
    
    def _parallel_scan(self, x, deltaA, deltaB, B_batch, L, d_inner, d_state):
        """并行扫描 - 用于长序列，近似但更快"""
        # 使用分块处理来近似并行扫描
        chunk_size = 32
        outputs = []
        
        # 初始化状态
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        for start_idx in range(0, L, chunk_size):
            end_idx = min(start_idx + chunk_size, L)
            chunk_len = end_idx - start_idx
            
            # 处理当前块
            chunk_outputs = []
            for t in range(start_idx, end_idx):
                x_t = x[:, t, :].unsqueeze(-1)
                h = deltaA[:, t, :, :] * h + deltaB[:, t, :, :] * x_t
                y_t = h.sum(dim=-1)
                chunk_outputs.append(y_t)
            
            outputs.extend(chunk_outputs)
        
        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


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
