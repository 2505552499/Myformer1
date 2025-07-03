"""
MambaEnhancedBlock - 替换DSTFormerBlock的Mamba增强版本
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .mamba_core import BidirectionalMamba, LocalMamba, GeometricReorder
from .mlp import MLP


class MambaEnhancedBlock(nn.Module):
    """
    Mamba-Enhanced Block: 替换DSTFormerBlock的核心时空建模
    基于PoseMamba的双向全局-局部状态空间模型
    
    设计理念:
    1. 全局建模: 双向Mamba处理长序列依赖
    2. 局部建模: 几何重排序 + 局部Mamba处理关节拓扑
    3. 自适应融合: 学习全局和局部特征的最优组合
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 n_frames=243, num_joints=17, d_state=16, d_conv=4, expand=2,
                 # 保持与DSTFormerBlock兼容的参数
                 num_heads=8, qkv_bias=False, qk_scale=None, hierarchical=False,
                 use_temporal_similarity=True, temporal_connection_len=1, 
                 use_tcn=False, graph_only=False, neighbour_num=4, attn_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.n_frames = n_frames
        self.num_joints = num_joints
        self.use_adaptive_fusion = use_adaptive_fusion
        self.hierarchical = hierarchical
        
        # 如果是hierarchical模式，调整维度
        if hierarchical:
            dim = dim // 2
            self.dim = dim
        
        # Mamba核心参数
        self.d_state = d_state  # SSM状态维度
        self.d_conv = d_conv    # 卷积核大小
        self.expand = expand    # 扩展因子
        self.d_inner = int(self.expand * dim)
        
        # 双向全局Mamba块 - 处理长序列时空依赖
        self.global_mamba = BidirectionalMamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 局部Mamba块 - 处理关节拓扑结构
        self.local_mamba = LocalMamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_joints=num_joints
        )
        
        # 几何重排序策略
        self.geometric_reorder = GeometricReorder(num_joints=num_joints)
        
        # 自适应融合机制
        if use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()
        
        # Layer Scale机制
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_local = nn.LayerNorm(dim)
        
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 用于调试和监控的标志
        self.debug_mode = False
    
    def _init_fusion(self):
        """初始化融合层权重"""
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)  # 初始时全局和局部各占50%
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: tensor with shape [B, T, J, C]
        Returns:
            tensor with shape [B, T, J, C]
        """
        B, T, J, C = x.shape
        
        if self.debug_mode:
            print(f"MambaEnhancedBlock input shape: {x.shape}")
        
        # 保存残差连接的输入
        residual = x
        
        # 全局时空建模 (双向Mamba)
        x_global = self.global_mamba(self.norm1(x))
        
        if self.debug_mode:
            print(f"Global Mamba output shape: {x_global.shape}")
        
        # 局部几何建模 (重排序 + 局部Mamba)
        x_reordered = self.geometric_reorder(self.norm_local(x))
        x_local = self.local_mamba(x_reordered)
        x_local = self.geometric_reorder.restore(x_local)  # 恢复原始顺序
        
        if self.debug_mode:
            print(f"Local Mamba output shape: {x_local.shape}")
        
        # 自适应融合
        if self.use_adaptive_fusion:
            # 拼接全局和局部特征
            fusion_input = torch.cat([x_global, x_local], dim=-1)  # [B, T, J, 2*C]
            
            # 计算融合权重
            alpha = self.fusion(fusion_input)  # [B, T, J, 2]
            alpha = alpha.softmax(dim=-1)
            
            # 加权融合
            x_mamba = x_global * alpha[..., 0:1] + x_local * alpha[..., 1:2]
            
            if self.debug_mode:
                print(f"Fusion weights - Global: {alpha[..., 0].mean():.3f}, Local: {alpha[..., 1].mean():.3f}")
        else:
            # 简单平均融合
            x_mamba = (x_global + x_local) / 2
        
        # 第一个残差连接 + Layer Scale
        if self.use_layer_scale:
            x = residual + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0).unsqueeze(0) * x_mamba
            )
        else:
            x = residual + self.drop_path(x_mamba)
        
        # MLP分支
        mlp_out = self.mlp(self.norm2(x))
        
        # 第二个残差连接 + Layer Scale
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0).unsqueeze(0) * mlp_out
            )
        else:
            x = x + self.drop_path(mlp_out)
        
        if self.debug_mode:
            print(f"MambaEnhancedBlock output shape: {x.shape}")
        
        return x
    
    def enable_debug(self):
        """启用调试模式"""
        self.debug_mode = True
        
    def disable_debug(self):
        """禁用调试模式"""
        self.debug_mode = False
    
    def get_complexity_info(self):
        """获取计算复杂度信息"""
        info = {
            'type': 'MambaEnhancedBlock',
            'dim': self.dim,
            'n_frames': self.n_frames,
            'num_joints': self.num_joints,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
            'use_adaptive_fusion': self.use_adaptive_fusion,
            'use_layer_scale': self.use_layer_scale
        }
        return info


def test_mamba_enhanced_block():
    """测试MambaEnhancedBlock的功能"""
    print("Testing MambaEnhancedBlock...")
    
    # 创建测试数据
    B, T, J, C = 2, 243, 17, 128
    x = torch.randn(B, T, J, C)
    
    # 创建模块
    block = MambaEnhancedBlock(
        dim=C,
        n_frames=T,
        num_joints=J,
        use_adaptive_fusion=True,
        use_layer_scale=True
    )
    
    # 启用调试模式
    block.enable_debug()
    
    # 前向传播
    with torch.no_grad():
        output = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shape match: {x.shape == output.shape}")
    
    # 检查梯度流
    block.train()
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    print("Gradient check passed!")
    print("MambaEnhancedBlock test completed successfully!")
    
    return True


if __name__ == "__main__":
    test_mamba_enhanced_block()
