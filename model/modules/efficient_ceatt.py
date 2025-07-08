import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CeATTForTCPFormer(nn.Module):
    """
    为TCPFormer优化的Compact-enhanced Attention机制
    基于GridFormer的CeATT设计，适配姿态序列数据
    """
    def __init__(self, dim, num_heads=8, temporal_sample_ratio=0.33, spatial_sample_ratio=0.5,
                 temporal_window=9, spatial_window=4, mode='temporal'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mode = mode  # 'temporal', 'spatial', 'both'
        self.scale = (dim // num_heads) ** -0.5

        # 确保head_dim是整数
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        if mode in ['temporal', 'both']:
            self.temporal_ceatt = TemporalCeATT(dim, num_heads, temporal_sample_ratio, temporal_window)

        if mode in ['spatial', 'both']:
            self.spatial_ceatt = SpatialCeATT(dim, num_heads, spatial_sample_ratio, spatial_window)

        if mode == 'both':
            self.fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)  # 与TCPFormer保持一致

    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - TCPFormer的标准输入格式
        Returns:
            output: [B, T, J, C] - 相同格式的输出
        """
        B, T, J, C = x.shape

        if self.mode == 'temporal':
            output = self.temporal_ceatt(x)
        elif self.mode == 'spatial':
            output = self.spatial_ceatt(x)
        else:  # both
            temp_out = self.temporal_ceatt(x)
            spat_out = self.spatial_ceatt(x)
            combined = torch.cat([temp_out, spat_out], dim=-1)
            output = self.fusion(combined)

        output = self.proj(output)
        output = self.dropout(output)

        return output


class TemporalCeATT(nn.Module):
    """
    时序维度的Compact-enhanced Attention
    处理每个关节在时间维度上的依赖关系
    """
    def __init__(self, dim, num_heads, sample_ratio, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sample_ratio = sample_ratio
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # 计算采样步长，确保合理的采样
        self.stride = max(1, int(1/sample_ratio))

        # 阶段1：时序采样器 - 使用平均池化更稳定
        self.temporal_sampler = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride)

        # 阶段2：紧凑自注意力
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=False)

        # 阶段3：局部增强 - 简化设计提高稳定性
        self.local_enhance = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        B, T, J, C = x.shape

        # 重排为 [B*J, C, T] 以便进行1D操作
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * J, C, T)

        # 阶段1：时序采样
        if self.stride > 1:
            x_sampled = self.temporal_sampler(x_reshaped)  # [B*J, C, T_sampled]
        else:
            x_sampled = x_reshaped
        T_sampled = x_sampled.shape[-1]

        # 阶段2：紧凑自注意力
        x_for_attn = x_sampled.permute(0, 2, 1)  # [B*J, T_sampled, C]
        qkv = self.qkv(x_for_attn).reshape(B * J, T_sampled, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # [B*J, T_sampled, num_heads, C//num_heads]

        q = q.permute(0, 2, 1, 3)  # [B*J, num_heads, T_sampled, C//num_heads]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_attn = (attn @ v).permute(0, 2, 1, 3).reshape(B * J, T_sampled, C)
        x_attn = self.attn_proj(x_attn)
        x_attn = x_attn.permute(0, 2, 1)  # [B*J, C, T_sampled]

        # 阶段3：局部增强
        x_enhanced = self.local_enhance(x_attn)

        # 上采样回原始时序长度
        if T_sampled != T:
            x_upsampled = F.interpolate(x_enhanced, size=T, mode='linear', align_corners=False)
        else:
            x_upsampled = x_enhanced

        # 重排回原始格式
        output = x_upsampled.reshape(B, J, C, T).permute(0, 3, 1, 2)  # [B, T, J, C]

        return output


class SpatialCeATT(nn.Module):
    """
    空间维度的Compact-enhanced Attention
    处理每个时间步内关节之间的空间关系
    """
    def __init__(self, dim, num_heads, sample_ratio, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sample_ratio = sample_ratio
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # 计算采样步长
        self.stride = max(1, int(1/sample_ratio))

        # 阶段1：空间采样器（关节采样）- 使用平均池化
        self.spatial_sampler = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride)

        # 阶段2：紧凑自注意力
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=False)

        # 阶段3：局部增强（关节邻域增强）
        self.local_enhance = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        B, T, J, C = x.shape

        # 重排为 [B*T, C, J] 以便进行1D操作
        x_reshaped = x.permute(0, 1, 3, 2).reshape(B * T, C, J)

        # 阶段1：空间采样
        if self.stride > 1:
            x_sampled = self.spatial_sampler(x_reshaped)  # [B*T, C, J_sampled]
        else:
            x_sampled = x_reshaped
        J_sampled = x_sampled.shape[-1]

        # 阶段2：紧凑自注意力
        x_for_attn = x_sampled.permute(0, 2, 1)  # [B*T, J_sampled, C]
        qkv = self.qkv(x_for_attn).reshape(B * T, J_sampled, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # [B*T, J_sampled, num_heads, C//num_heads]

        q = q.permute(0, 2, 1, 3)  # [B*T, num_heads, J_sampled, C//num_heads]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_attn = (attn @ v).permute(0, 2, 1, 3).reshape(B * T, J_sampled, C)
        x_attn = self.attn_proj(x_attn)
        x_attn = x_attn.permute(0, 2, 1)  # [B*T, C, J_sampled]

        # 阶段3：局部增强
        x_enhanced = self.local_enhance(x_attn)

        # 上采样回原始关节数
        if J_sampled != J:
            x_upsampled = F.interpolate(x_enhanced, size=J, mode='linear', align_corners=False)
        else:
            x_upsampled = x_enhanced

        # 重排回原始格式
        output = x_upsampled.reshape(B, T, C, J).permute(0, 1, 3, 2)  # [B, T, J, C]

        return output


class CeATTEnhancedAttention(nn.Module):
    """
    替换TCPFormer原始Attention模块的CeATT版本
    保持接口兼容性
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., mode='spatial', vis='no'):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.vis = vis
        self.scale = qk_scale or (dim_in // num_heads) ** -0.5

        # 确保维度兼容性
        assert dim_in % num_heads == 0, f"dim_in {dim_in} must be divisible by num_heads {num_heads}"

        # 使用CeATT替换传统注意力
        if mode == 'temporal':
            self.ceatt = CeATTForTCPFormer(
                dim_in, num_heads,
                temporal_sample_ratio=0.33,
                spatial_sample_ratio=0.5,
                mode='temporal'
            )
        elif mode == 'spatial':
            self.ceatt = CeATTForTCPFormer(
                dim_in, num_heads,
                temporal_sample_ratio=0.33,
                spatial_sample_ratio=0.5,
                mode='spatial'
            )
        else:
            self.ceatt = CeATTForTCPFormer(
                dim_in, num_heads,
                temporal_sample_ratio=0.33,
                spatial_sample_ratio=0.5,
                mode='both'
            )

        self.proj = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        保持与原始Attention模块相同的接口
        """
        # 应用CeATT
        x = self.ceatt(x)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
