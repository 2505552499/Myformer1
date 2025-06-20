from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath
import os,sys
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.modules.attention import Attention, FrequencyAwareAttention
from model.modules.mlp import MLP
from model.modules.crossattention import CrossAttention, FrequencyAwareCrossAttention
from model.modules.sum_attention import Sum_Attention, FrequencyAwareSumAttention



class MIBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='temporal', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243,is_local = None,
                 use_frequency_aware=True, freq_ratio=0.5):
        super().__init__()
        self.norm_full = nn.LayerNorm(dim)
        self.norm_center = nn.LayerNorm(dim)
        self.use_frequency_aware = use_frequency_aware
        mlp_hidden_dim = int(dim*mlp_ratio)

        # Choose attention type based on frequency awareness
        cross_attention_class = FrequencyAwareCrossAttention if use_frequency_aware else CrossAttention
        attention_class = FrequencyAwareAttention if use_frequency_aware else Attention
        sum_attention_class = FrequencyAwareSumAttention if use_frequency_aware else Sum_Attention

        cross_attention_kwargs = {'freq_ratio': freq_ratio, 'use_low_freq': True} if use_frequency_aware else {}
        attention_kwargs = {'freq_ratio': freq_ratio, 'use_low_freq': True} if use_frequency_aware else {}
        sum_attention_kwargs = {'freq_ratio': freq_ratio, 'use_low_freq': True} if use_frequency_aware else {}

        self.full_center = cross_attention_class(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,back_att=True, **cross_attention_kwargs)
        self.center_full = cross_attention_class(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,back_att=True, **cross_attention_kwargs)
        self.mlp_1 = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.mlp_2 = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_4 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_5 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_6 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_7 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_8 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)


        self.norm_sa_self = nn.LayerNorm(dim)
        self.map_sa_self = attention_class(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode, **attention_kwargs)
        self.norm_mlp_self = nn.LayerNorm(dim)
        self.mlp_sa_self = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

        self.norm_sa_1 = nn.LayerNorm(dim)
        self.map_sum = sum_attention_class(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode, **sum_attention_kwargs)
        self.norm_sa_2 = nn.LayerNorm(dim)
        self.mlp_sa = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

        
        self.sg = nn.Sigmoid()
        self.att_weight = nn.Parameter(torch.rand(1))

    def forward(self, x,pose_query):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:

            attn_map_1,out_1 = self.center_full(self.norm_center(pose_query),self.norm_full(x))
            pose_query = pose_query + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)*out_1)
            pose_query = pose_query + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0)*self.mlp_1(self.norm_1(pose_query)))

            attn_map_2,out_2 = self.full_center(self.norm_full(x),self.norm_center(pose_query))
            x = x + self.drop_path(self.layer_scale_3.unsqueeze(0).unsqueeze(0)*out_2)
            x = x + self.drop_path(self.layer_scale_4.unsqueeze(0).unsqueeze(0)*self.mlp_2(self.norm_2(x)))

            attn_map = attn_map_2 @ attn_map_1

            norm_weight = self.sg(self.att_weight)

            x = x + self.drop_path(self.layer_scale_7.unsqueeze(0).unsqueeze(1)*self.map_sa_self(self.norm_sa_self(x)))
            x = x + self.drop_path(self.layer_scale_8.unsqueeze(0).unsqueeze(1)*self.mlp_sa_self(self.norm_mlp_self(x)))

            x = x + self.drop_path(self.layer_scale_5.unsqueeze(0).unsqueeze(1)*self.map_sum(self.norm_sa_1(x),attn_map,norm_weight))
            x = x + self.drop_path(self.layer_scale_6.unsqueeze(0).unsqueeze(1)*self.mlp_sa(self.norm_sa_2(x)))

        return x,pose_query
    




