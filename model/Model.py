from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath
import os,sys
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules.attention import Attention, FrequencyAwareAttention
from model.modules.mlp import MLP
from model.modules.crossattention import CrossAttention
from model.modules.ModelBlock import MIBlock
from model.modules.joint_flow import JointFlow
from model.modules.enhanced_motion_flow import EnhancedMotionFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


class TransBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer_type = mixer_type
        if mixer_type == 'crossattention': 
            self.local_attention_list = nn.ModuleList([
                Attention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode) for i in range(3)
            ])
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.loacl_mlps =  nn.ModuleList([
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop) for i in range(3)
            ])
            self.len = 0
            self.normq = nn.LayerNorm(dim)
            self.normkv = nn.LayerNorm(dim)
            self.mixer = nn.ModuleList([
                CrossAttention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode),
                CrossAttention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode),
                CrossAttention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode),
            ])
            self.self_attention = Attention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,vis='yes')
            self.mlps = nn.ModuleList([
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop),
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop),
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
            ])
            self.sa_mlp = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
            self.norms = nn.ModuleList([
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim)
            ])
        elif mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)


        


    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """

        if self.mixer_type == 'crossattention':
            x = self.forward_local(x)
            self.len = x.shape[1] // 3
            x = self.forward_cross(x,self.len)
            return x
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def forward_cross(self,x,len):
        part_size = len
        first_part = x[:,:part_size]
        middel_part = x[:,part_size:2*part_size]
        last_part = x[:,2*part_size:]
        q = []
        kv = []
        q.append(first_part)
        q.append(middel_part)
        q.append(last_part)
        kv.append(torch.cat([middel_part,last_part],dim=1))
        kv.append(torch.cat([first_part,last_part],dim=1))
        kv.append(torch.cat([middel_part,last_part],dim=1))

        for i in range(3):
            if self.use_layer_scale:
                q[i] = q[i] + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)*self.mixer[i](self.normq(q[i]),self.normkv(kv[i])))
                q[i] = q[i] + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)*self.mlps[i](self.norms[i](q[i])))

            else:
                q[i] = q[i] + self.drop_path(self.mixer[i](self.normq(q[i]),self.normkv(kv[i])))
                q[i] = q[i] + self.drop_path(self.mlps[i](self.norms[i](q[i])))

        out = torch.cat(q,dim=1)
        if self.use_layer_scale:
            out = out + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.self_attention(self.norm1(out)))
            out = out + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.sa_mlp(self.norm2(out)))
        else:
            out = out + self.drop_path(self.self_attention(self.norm1(out)))
            out = out + self.drop_path(self.sa_mlp(self.norm2(out)))

        return out

    def forward_local(self,x):
        x = list(torch.chunk(x,3,dim=1))

        for i in range(3):
            if self.use_layer_scale:
                x[i] = x[i] + self.drop_path(
                    self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                    * self.local_attention_list[i](self.norm1(x[i])))
                x[i] = x[i] + self.drop_path(
                    self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                    * self.loacl_mlps[i](self.norm2(x[i])))
            else:
                x[i] = x[i] + self.drop_path(self.local_attention_list[i](self.norm1(x[i])))
                x[i] = x[i] + self.drop_path(self.loacl_mlps[i](self.norm2(x[i])))

        out = torch.cat(x,dim=1)
        return out


class PoseFormerV2EnhancedBlock(nn.Module):
    """
    PoseFormerV2-inspired bidirectional frequency-enhanced block
    Forward branch: Temporal-Spatial + Frequency-Temporal
    Backward branch: Frequency-Temporal + Temporal-Spatial
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243, freq_ratio=0.5):
        super().__init__()
        self.mode = mode
        self.freq_ratio = freq_ratio

        # Forward branch: Temporal-Spatial → Frequency-Temporal
        self.forward_temporal_spatial = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                                proj_drop=drop, mode='temporal')
        self.forward_spatial_norm = nn.LayerNorm(dim)
        self.forward_spatial = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                       proj_drop=drop, mode='spatial')

        self.forward_freq_temporal = FrequencyAwareAttention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                                           proj_drop=drop, mode='temporal', freq_ratio=freq_ratio)
        self.forward_freq_norm = nn.LayerNorm(dim)

        # Backward branch: Frequency-Temporal → Temporal-Spatial
        self.backward_freq_temporal = FrequencyAwareAttention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                                            proj_drop=drop, mode='temporal', freq_ratio=freq_ratio)
        self.backward_freq_norm = nn.LayerNorm(dim)

        self.backward_temporal_spatial = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                                 proj_drop=drop, mode='temporal')
        self.backward_spatial_norm = nn.LayerNorm(dim)
        self.backward_spatial = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                        proj_drop=drop, mode='spatial')

        # Branch fusion
        self.branch_fusion = nn.Linear(dim * 2, dim)
        self.fusion_gate = nn.Parameter(torch.ones(1))

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_forward = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_backward = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_mlp = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        residual = x

        # Forward branch: Temporal-Spatial → Frequency-Temporal
        forward_out = x
        # Step 1: Temporal-Spatial attention
        forward_out = forward_out + self.drop_path(self.forward_temporal_spatial(forward_out))
        forward_out = forward_out + self.drop_path(self.forward_spatial(self.forward_spatial_norm(forward_out)))

        # Step 2: Frequency-Temporal attention
        forward_out = forward_out + self.drop_path(self.forward_freq_temporal(self.forward_freq_norm(forward_out)))

        # Backward branch: Frequency-Temporal → Temporal-Spatial
        backward_out = x
        # Step 1: Frequency-Temporal attention
        backward_out = backward_out + self.drop_path(self.backward_freq_temporal(self.backward_freq_norm(backward_out)))

        # Step 2: Temporal-Spatial attention
        backward_out = backward_out + self.drop_path(self.backward_temporal_spatial(backward_out))
        backward_out = backward_out + self.drop_path(self.backward_spatial(self.backward_spatial_norm(backward_out)))

        # Fuse forward and backward branches
        branch_combined = torch.cat([forward_out, backward_out], dim=-1)
        branch_fused = self.branch_fusion(branch_combined)

        # Apply gating and layer scaling
        gate = torch.sigmoid(self.fusion_gate)
        if self.use_layer_scale:
            branch_output = self.drop_path(
                self.layer_scale_forward.unsqueeze(0).unsqueeze(0) * gate * branch_fused +
                self.layer_scale_backward.unsqueeze(0).unsqueeze(0) * (1 - gate) * forward_out
            )
        else:
            branch_output = self.drop_path(gate * branch_fused + (1 - gate) * forward_out)

        x = residual + branch_output

        # MLP path
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_mlp.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm_mlp(x))
            )
        else:
            x = x + self.drop_path(self.mlp(self.norm_mlp(x)))

        return x


class DSTFormerBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,
                 use_frequency_enhanced=True, freq_ratio=0.5):
        super().__init__()
        self.hierarchical = hierarchical
        self.use_frequency_enhanced = use_frequency_enhanced
        dim = dim // 2 if hierarchical else dim

        # Choose block type based on frequency enhancement
        block_class = PoseFormerV2EnhancedBlock if use_frequency_enhanced else TransBlock
        block_kwargs = {'freq_ratio': freq_ratio} if use_frequency_enhanced else {}

        self.att_spatial = block_class(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames, **block_kwargs)
        self.att_temporal = block_class(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames, **block_kwargs)



        self.graph_spatial = block_class(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='temporal', mixer_type="attention",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames, **block_kwargs)
        self.graph_temporal = block_class(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='spatial', mixer_type='attention',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames, **block_kwargs)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """

        x_attn = self.att_temporal(self.att_spatial(x))
        x_graph = self.graph_temporal(self.graph_spatial(x))


        alpha = torch.cat((x_attn, x_graph), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]

        return x


class MemoryInducedBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,mode='temporal',
                 use_frequency_aware=True, freq_ratio=0.5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_frequency_aware = use_frequency_aware
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Choose attention type based on frequency awareness
        attention_class = FrequencyAwareAttention if use_frequency_aware else Attention
        attention_kwargs = {'freq_ratio': freq_ratio, 'use_low_freq': True} if use_frequency_aware else {}

        self.local_attention_list = nn.ModuleList([
                attention_class(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode, **attention_kwargs) for i in range(3)
            ])
        self.loacl_mlps =  nn.ModuleList([
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop) for i in range(3)
            ])
        self.layer_scale = nn.ParameterList([
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        ])
        self.local_norms = nn.ModuleList([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim)
        ])

        self.cross_temporal = MIBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                          use_frequency_aware=use_frequency_aware,
                                          freq_ratio=freq_ratio)

    def forward(self, x,pose_query):
        """
        x: tensor with shape [B, T, J, C]
        """
        x = list(torch.chunk(x,3,dim=1))

        for i in range(3):
            x[i] = x[i] + self.drop_path(self.layer_scale[i].unsqueeze(0).unsqueeze(0) * self.local_attention_list[i](self.local_norms[i](x[i])))
            x[i] = x[i] + self.drop_path(self.layer_scale[i+3].unsqueeze(0).unsqueeze(0) * self.loacl_mlps[i](self.local_norms[i+3](x[i])))

        x = torch.cat(x,dim=1)

        x,pose_query = self.cross_temporal(x,pose_query)

        return x,pose_query









def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,type = None,
                  use_frequency_aware=True, freq_ratio=0.5, use_frequency_enhanced=True):

    layers = []
    for _ in range(n_layers):
        if type == 'temporal':
            layers.append(MemoryInducedBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                          use_frequency_aware=use_frequency_aware,
                                          freq_ratio=freq_ratio))
        else:
            layers.append(DSTFormerBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                          use_frequency_enhanced=use_frequency_enhanced,
                                          freq_ratio=freq_ratio))
    layers = nn.Sequential(*layers)

    return layers


class MemoryInducedTransformer(nn.Module):


    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243, use_joint_flow=True, motion_dim=64, joint_flow_dropout=0.1,
                 use_enhanced_motion=False, temporal_scales=[1, 8, 25, 100], motion_output_high_dim=True,
                 use_frequency_aware=True, freq_ratio=0.5, use_frequency_enhanced=True):

        super().__init__()

        # Motion encoding module
        self.use_joint_flow = use_joint_flow
        self.use_enhanced_motion = use_enhanced_motion
        self.motion_output_high_dim = motion_output_high_dim

        if use_enhanced_motion:
            # Use enhanced motion flow with multi-scale temporal modeling
            self.motion_flow = EnhancedMotionFlow(
                dim_in=dim_in,
                motion_dim=motion_dim,
                dropout=joint_flow_dropout,
                temporal_scales=temporal_scales,
                output_high_dim=motion_output_high_dim
            )
        elif use_joint_flow:
            # Use basic joint flow
            self.joint_flow = JointFlow(
                dim_in=dim_in,
                motion_dim=motion_dim,
                use_velocity=True,
                use_acceleration=True,
                dropout=joint_flow_dropout
            )

        # Embedding - adapt to motion flow output dimension
        if use_enhanced_motion and motion_output_high_dim:
            self.joints_embed = nn.Linear(motion_dim, dim_feat)
        else:
            self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)
        self.layers_num = n_layers
        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames,
                                    use_frequency_aware=use_frequency_aware,
                                    freq_ratio=freq_ratio,
                                    use_frequency_enhanced=use_frequency_enhanced)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

        self.temporal_layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames,
                                    type='temporal',
                                    use_frequency_aware=use_frequency_aware,
                                    freq_ratio=freq_ratio,
                                    use_frequency_enhanced=use_frequency_enhanced)
        

        self.center_pose = nn.Parameter(torch.randn(int(n_frames/3),num_joints,dim_feat))
        self.center_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        """
        b,t,j,c = x.shape

        # Apply motion enhancement at input layer
        if self.use_enhanced_motion:
            x = self.motion_flow(x)
        elif self.use_joint_flow:
            x = self.joint_flow(x)

        pose_query = self.center_pose.unsqueeze(0).repeat(b,1,1,1)
        pose_query = pose_query + self.center_pos_embed
        x = self.joints_embed(x)  #
        x = x + self.pos_embed

        for layer,temporal_layer in zip(self.layers,self.temporal_layers):
            x = layer(x)
            x,pose_query = temporal_layer(x,pose_query)


        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x


class OptimizedTCPFormer(nn.Module):
    """
    Optimized TCPFormer with frequency-enhanced DSTFormer and frequency-aware MemoryInduced blocks
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243, use_joint_flow=True, motion_dim=64, joint_flow_dropout=0.1,
                 use_enhanced_motion=False, temporal_scales=[1, 8, 25, 100], motion_output_high_dim=True,
                 freq_ratio=0.5, dst_freq_ratio=0.3, memory_freq_ratio=0.7):

        super().__init__()

        # Motion encoding module
        self.use_joint_flow = use_joint_flow
        self.use_enhanced_motion = use_enhanced_motion
        self.motion_output_high_dim = motion_output_high_dim

        if use_enhanced_motion:
            self.motion_flow = EnhancedMotionFlow(
                dim_in=dim_in,
                motion_dim=motion_dim,
                dropout=joint_flow_dropout,
                temporal_scales=temporal_scales,
                output_high_dim=motion_output_high_dim
            )
        elif use_joint_flow:
            self.joint_flow = JointFlow(
                dim_in=dim_in,
                motion_dim=motion_dim,
                use_velocity=True,
                use_acceleration=True,
                dropout=joint_flow_dropout
            )

        # Embedding
        if use_enhanced_motion and motion_output_high_dim:
            self.joints_embed = nn.Linear(motion_dim, dim_feat)
        else:
            self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        # Frequency-enhanced DSTFormer layers
        self.freq_dst_layers = create_layers(
            dim=dim_feat, n_layers=n_layers, mlp_ratio=mlp_ratio, act_layer=act_layer,
            attn_drop=attn_drop, drop_rate=drop, drop_path_rate=drop_path,
            num_heads=num_heads, use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias, qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion, hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn, graph_only=graph_only,
            neighbour_num=neighbour_num, n_frames=n_frames,
            type=None,  # DSTFormer layers
            use_frequency_aware=False,  # DSTFormer uses frequency_enhanced instead
            freq_ratio=dst_freq_ratio,
            use_frequency_enhanced=True
        )

        # Frequency-aware MemoryInduced layers
        self.freq_aware_memory_layers = create_layers(
            dim=dim_feat, n_layers=n_layers, mlp_ratio=mlp_ratio, act_layer=act_layer,
            attn_drop=attn_drop, drop_rate=drop, drop_path_rate=drop_path,
            num_heads=num_heads, use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias, qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion, hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn, graph_only=graph_only,
            neighbour_num=neighbour_num, n_frames=n_frames,
            type='temporal',  # MemoryInduced layers
            use_frequency_aware=True,
            freq_ratio=memory_freq_ratio,
            use_frequency_enhanced=False  # MemoryInduced uses frequency_aware instead
        )

        # Output layers
        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

        # Pose query for MemoryInduced blocks
        self.center_pose = nn.Parameter(torch.randn(int(n_frames/3), num_joints, dim_feat))
        self.center_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))

    def initialize_pose_query(self, x):
        """Initialize pose query for MemoryInduced blocks"""
        b = x.shape[0]
        pose_query = self.center_pose.unsqueeze(0).repeat(b, 1, 1, 1)
        pose_query = pose_query + self.center_pos_embed
        return pose_query

    def forward(self, x, return_rep=False):
        """
        Forward pass of OptimizedTCPFormer
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        """
        b, t, j, c = x.shape

        # Apply motion enhancement at input layer
        if self.use_enhanced_motion:
            x = self.motion_flow(x)
        elif self.use_joint_flow:
            x = self.joint_flow(x)

        # Initialize pose query
        pose_query = self.initialize_pose_query(x)

        # Embedding
        x = self.joints_embed(x)
        x = x + self.pos_embed

        # Alternating frequency-enhanced DSTFormer and frequency-aware MemoryInduced
        for i in range(len(self.freq_dst_layers)):
            # Frequency-enhanced DSTFormer
            x = self.freq_dst_layers[i](x)

            # Frequency-aware MemoryInduced
            x, pose_query = self.freq_aware_memory_layers[i](x, pose_query)

        # Output processing
        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)
        return x


def _test():
    torch.cuda.set_device(0)
    # from torchprofile import profile_macs  # Comment out if not available
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = MemoryInducedTransformer(n_layers=16, dim_in=3, dim_feat=128, mlp_ratio=4, hierarchical=False,
                           use_tcn=False, graph_only=False, n_frames=t, use_joint_flow=True, motion_dim=32).to('cuda')
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    # print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # # Warm-up to avoid timing fluctuations
    # for _ in range(10):
    #     _ = model(random_x)

    # import time
    # num_iterations = 100 
    # # Measure the inference time for 'num_iterations' iterations
    # start_time = time.time()
    # for _ in range(num_iterations):
    #     with torch.no_grad():
    #         _ = model(random_x)
    # end_time = time.time()

    # # Calculate the average inference time per iteration
    # average_inference_time = (end_time - start_time) / num_iterations

    # # Calculate FPS
    # fps = 1.0 / average_inference_time

    # print(f"FPS: {fps}")
    

    out = model(random_x)

    # assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
