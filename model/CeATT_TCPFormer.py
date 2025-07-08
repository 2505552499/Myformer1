import torch
import torch.nn as nn
import sys
import os

# 添加路径以确保导入正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.Model import MemoryInducedTransformer, DSTFormerBlock, MemoryInducedBlock, create_layers
from model.modules.efficient_ceatt import CeATTEnhancedAttention, CeATTForTCPFormer


class CeATTEnhancedTCPFormer(MemoryInducedTransformer):
    """
    集成CeATT机制的TCPFormer
    在关键位置替换传统注意力为CeATT
    """
    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4,
                 act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, use_adaptive_fusion=True, num_heads=4,
                 qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                 graph_only=False, neighbour_num=4, n_frames=243,
                 # CeATT特定参数
                 ceatt_config=None):

        # 先保存参数，确保在替换过程中可以访问
        self.dim_feat = dim_feat
        self.num_heads = num_heads

        # CeATT配置
        self.ceatt_config = ceatt_config or {
            'temporal_sample_ratio': 0.33,
            'spatial_sample_ratio': 0.5,
            'temporal_window': 9,
            'spatial_window': 4,
            'replace_strategy': 'progressive'  # 'progressive', 'full', 'selective'
        }

        # 初始化基础TCPFormer
        super().__init__(n_layers, dim_in, dim_feat, dim_rep, dim_out, mlp_ratio,
                        act_layer, attn_drop, drop, drop_path, use_layer_scale,
                        layer_scale_init_value, use_adaptive_fusion, num_heads,
                        qkv_bias, qkv_scale, hierarchical, num_joints,
                        use_temporal_similarity, temporal_connection_len, use_tcn,
                        graph_only, neighbour_num, n_frames)

        # 应用CeATT替换策略
        self._apply_ceatt_replacement()
        
    def _apply_ceatt_replacement(self):
        """
        根据配置策略替换注意力模块
        """
        strategy = self.ceatt_config.get('replace_strategy', 'progressive')

        if strategy == 'progressive':
            self._progressive_replacement()
        elif strategy == 'full':
            self._full_replacement()
        elif strategy == 'selective':
            self._selective_replacement()

    def _progressive_replacement(self):
        """
        渐进式替换：优先替换计算量最大的模块
        """
        print(f"Applying CeATT progressive replacement to {len(self.layers)} DSTFormer layers...")

        # 1. 替换DSTFormerBlock中的时序注意力（最大瓶颈）
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'att_temporal'):
                print(f"Replacing temporal attention in layer {i}")
                layer.att_temporal = CeATTEnhancedAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    attn_drop=0.,
                    proj_drop=0.,
                    mode='temporal'
                )

        print(f"Applying CeATT to {len(self.temporal_layers)} MemoryInduced layers...")

        # 2. 替换MemoryInducedBlock中的交叉注意力
        for i, layer in enumerate(self.temporal_layers):
            if hasattr(layer, 'center_full'):
                print(f"Replacing center_full attention in temporal layer {i}")
                layer.center_full = CeATTEnhancedCrossAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    mode='temporal'
                )
            if hasattr(layer, 'full_center'):
                print(f"Replacing full_center attention in temporal layer {i}")
                layer.full_center = CeATTEnhancedCrossAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    mode='temporal'
                )

    def _full_replacement(self):
        """
        全面替换：替换所有注意力模块
        """
        print("Applying CeATT full replacement...")

        # 替换所有DSTFormerBlock
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'att_spatial'):
                print(f"Replacing spatial attention in layer {i}")
                layer.att_spatial = CeATTEnhancedAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    attn_drop=0.,
                    proj_drop=0.,
                    mode='spatial'
                )
            if hasattr(layer, 'att_temporal'):
                print(f"Replacing temporal attention in layer {i}")
                layer.att_temporal = CeATTEnhancedAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    attn_drop=0.,
                    proj_drop=0.,
                    mode='temporal'
                )

        # 替换所有MemoryInducedBlock
        self._progressive_replacement()

    def _selective_replacement(self):
        """
        选择性替换：仅替换特定层
        """
        print("Applying CeATT selective replacement...")

        # 仅替换后半部分层（更深层的特征更抽象）
        mid_point = len(self.layers) // 2

        for i, layer in enumerate(self.layers[mid_point:], mid_point):
            if hasattr(layer, 'att_temporal'):
                print(f"Replacing temporal attention in layer {i}")
                layer.att_temporal = CeATTEnhancedAttention(
                    dim_in=self.dim_feat,
                    dim_out=self.dim_feat,
                    num_heads=self.num_heads,
                    attn_drop=0.,
                    proj_drop=0.,
                    mode='temporal'
                )


class CeATTEnhancedCrossAttention(nn.Module):
    """
    CeATT增强的交叉注意力模块
    用于MemoryInducedBlock中的center_full和full_center
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qkv_scale=None,
                 attn_drop=0., proj_drop=0., mode='temporal', back_att=True):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.back_att = back_att  # 默认为True，与原始CrossAttention保持一致
        self.scale = qkv_scale or (dim_in // num_heads) ** -0.5

        # 确保维度兼容性
        assert dim_in % num_heads == 0, f"dim_in {dim_in} must be divisible by num_heads {num_heads}"

        # 简化设计：直接使用线性层，不使用CeATT预处理
        # 这样可以避免维度不匹配的问题
        self.wq = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.wk = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.wv = nn.Linear(dim_in, dim_in, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        # CeATT增强模块（可选）
        self.use_ceatt_enhance = False  # 暂时禁用以确保稳定性
        if self.use_ceatt_enhance:
            self.q_enhance = CeATTForTCPFormer(dim_in, num_heads, mode='temporal')
            self.kv_enhance = CeATTForTCPFormer(dim_in, num_heads, mode='temporal')

    def forward(self, q, kv):
        """
        Args:
            q: [B, T_q, J, C] - 查询序列
            kv: [B, T_kv, J, C] - 键值序列
        """
        b, t_q, j, d = q.shape
        t_kv = kv.shape[1]

        # 可选的CeATT增强
        if self.use_ceatt_enhance:
            q = self.q_enhance(q)
            kv = self.kv_enhance(kv)

        # 标准交叉注意力计算
        q_proj = self.wq(q).reshape(b, t_q, j, self.num_heads, d // self.num_heads).permute(0, 3, 2, 1, 4)
        k_proj = self.wk(kv).reshape(b, t_kv, j, self.num_heads, d // self.num_heads).permute(0, 3, 2, 1, 4)
        v_proj = self.wv(kv).reshape(b, t_kv, j, self.num_heads, d // self.num_heads).permute(0, 3, 2, 1, 4)

        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_proj  # [b, h, j, t_q, c]
        out = out.permute(0, 3, 2, 1, 4).reshape(b, t_q, j, d)
        out = self.proj(out)
        out = self.proj_drop(out)

        if self.back_att:
            return attn, out
        else:
            return out


def create_ceatt_tcpformer(config):
    """
    创建CeATT增强的TCPFormer模型
    """
    ceatt_config = {
        'temporal_sample_ratio': config.get('temporal_sample_ratio', 0.33),
        'spatial_sample_ratio': config.get('spatial_sample_ratio', 0.5),
        'temporal_window': config.get('temporal_window', 9),
        'spatial_window': config.get('spatial_window', 4),
        'replace_strategy': config.get('replace_strategy', 'progressive')
    }

    model = CeATTEnhancedTCPFormer(
        n_layers=config.get('n_layers', 16),
        dim_in=config.get('dim_in', 3),
        dim_feat=config.get('dim_feat', 128),
        dim_rep=config.get('dim_rep', 512),
        dim_out=config.get('dim_out', 3),
        mlp_ratio=config.get('mlp_ratio', 4),
        act_layer=nn.GELU,
        attn_drop=config.get('attn_drop', 0.0),
        drop=config.get('drop', 0.0),
        drop_path=config.get('drop_path', 0.0),
        use_layer_scale=config.get('use_layer_scale', True),
        layer_scale_init_value=config.get('layer_scale_init_value', 1e-5),
        use_adaptive_fusion=config.get('use_adaptive_fusion', True),
        num_heads=config.get('num_heads', 8),
        qkv_bias=config.get('qkv_bias', False),
        qkv_scale=config.get('qkv_scale', None),
        hierarchical=config.get('hierarchical', False),
        num_joints=config.get('num_joints', 17),
        use_temporal_similarity=config.get('use_temporal_similarity', True),
        temporal_connection_len=config.get('temporal_connection_len', 1),
        use_tcn=config.get('use_tcn', False),
        graph_only=config.get('graph_only', False),
        neighbour_num=config.get('neighbour_num', 4),
        n_frames=config.get('n_frames', 243),
        ceatt_config=ceatt_config
    )

    return model


# 使用示例和测试
if __name__ == "__main__":
    # 配置参数
    config = {
        'n_layers': 16,
        'dim_in': 3,
        'dim_feat': 128,
        'num_heads': 8,
        'n_frames': 243,
        'temporal_sample_ratio': 0.33,
        'spatial_sample_ratio': 0.5,
        'replace_strategy': 'progressive'
    }

    print("Creating CeATT-TCPFormer model...")
    model = create_ceatt_tcpformer(config)

    # 测试输入
    batch_size = 2
    n_frames = 243
    n_joints = 17
    input_dim = 3

    x = torch.randn(batch_size, n_frames, n_joints, input_dim)
    print(f"Input shape: {x.shape}")

    # 前向传播测试
    try:
        with torch.no_grad():
            output = model(x)
            print(f"Output shape: {output.shape}")
            print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
