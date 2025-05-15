import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Model import TransBlock as TransformerBlock
from model.Model import MemoryInducedBlock as TemporalTransformerBlock

# 将之前实现的所有改进类复制到这个文件中
# HourglassTokenizer, GlobalLocalCommunicationModule, BiomechanicalConstraintModule 等

# 然后创建改进版的MemoryInducedTransformer

class HourglassTokenizer(nn.Module):
    """
    HourglassTokenizer类实现对输入序列的时间维度压缩和扩展，
    允许模型根据动作复杂度动态选择重要的时间步。
    """
    def __init__(self, dim, num_frames=243, num_joints=17, reduction_ratio=4, min_tokens=16):
        super().__init__()
        self.dim = dim                    # 特征维度
        self.num_frames = num_frames      # 输入序列长度
        self.num_joints = num_joints      # 关节数量
        self.min_tokens = min_tokens      # 最小令牌数
        self.reduction_ratio = reduction_ratio  # 时间维度压缩比例
        
        # 特征处理层
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # 令牌重要性预测器
        self.token_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 输出处理层
        self.output_processor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
    def forward(self, x, complexity_factor=None):
        """
        前向传播
        Args:
            x: 形状为[B, T, J, C]的输入张量
            complexity_factor: 动作复杂度因子，用于动态确定要保留的令牌数
        Returns:
            形状为[B, T, J, C]的输出张量
        """
        B, T, J, C = x.shape
        identity = x  # 保存输入用于残差连接
        
        # 1. 特征处理
        x = self.feature_processor(x)  # [B, T, J, C]
        
        # 2. 时间维度下采样 - 使用自适应平均池化
        # 计算目标下采样长度
        T_reduced = max(self.min_tokens, T // self.reduction_ratio)
        
        # 重新排列维度以便在时间维度上执行池化
        x_reshaped = x.permute(0, 2, 3, 1)  # [B, J, C, T]
        x_reshaped = x_reshaped.reshape(B*J*C, 1, T)  # [B*J*C, 1, T]
        
        # 应用自适应平均池化
        x_pooled = F.adaptive_avg_pool1d(x_reshaped, T_reduced)  # [B*J*C, 1, T_reduced]
        
        # 恢复原始维度顺序
        x_pooled = x_pooled.reshape(B, J, C, T_reduced)  # [B, J, C, T_reduced]
        x_downsampled = x_pooled.permute(0, 3, 1, 2)  # [B, T_reduced, J, C]
        
        # 3. 自适应令牌选择（如果需要）
        if complexity_factor is not None and complexity_factor < 1.0 and T_reduced > self.min_tokens:
            # 计算每个时间步的重要性分数
            token_importance = self.token_scorer(x_downsampled.mean(dim=2))  # [B, T_reduced, 1]
            token_importance = token_importance.squeeze(-1)  # [B, T_reduced]
            
            # 基于复杂度因子选择要保留的令牌数
            k = max(self.min_tokens, int(T_reduced * complexity_factor))
            
            # 选择最重要的k个时间步
            _, indices = torch.topk(token_importance, k=k, dim=1)  # [B, k]
            indices, _ = torch.sort(indices, dim=1)  # 保持时间顺序
            
            # 批量索引
            batch_indices = torch.arange(B, device=x.device).view(-1, 1).expand(-1, k)  # [B, k]
            
            # 收集重要的时间步
            x_selected = x_downsampled[batch_indices, indices]  # [B, k, J, C]
            T_selected = k
        else:
            # 不进行选择，保留所有时间步
            x_selected = x_downsampled
            T_selected = T_reduced
        
        # 4. 时间维度上采样回原始长度
        x_to_upsample = x_selected.permute(0, 2, 3, 1)  # [B, J, C, T_selected]
        
        # 调整维度以便正确上采样
        x_to_upsample = x_to_upsample.reshape(B*J*C, 1, T_selected)  # [B*J*C, 1, T_selected]
        
        # 使用线性插值进行上采样
        x_upsampled = F.interpolate(
            x_to_upsample, 
            size=T, 
            mode='linear', 
            align_corners=False
        )  # [B*J*C, 1, T]
        
        # 恢复原始维度
        x_upsampled = x_upsampled.reshape(B, J, C, T)  # [B, J, C, T]
        x_upsampled = x_upsampled.permute(0, 3, 1, 2)  # [B, T, J, C]
        
        # 5. 输出处理和残差连接
        x_output = self.output_processor(x_upsampled)
        
        return x_output + identity  # [B, T, J, C]

class GlobalLocalCommunicationModule(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 全局特征提取器
        self.global_pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # 局部特征增强器
        self.local_enhancer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 跨尺度注意力
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        
        # 融合门控单元
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 特征转换
        self.global_transform = nn.Linear(dim, dim)
        self.local_transform = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x, initial_features=None):
        # x: [B, T, J, C]
        B, T, J, C = x.shape
        identity = x
        
        # 提取全局特征 - 使用平均池化聚合关节维度
        x_global = x.mean(dim=2)  # [B, T, C]
        x_global = self.global_pool(x_global)  # [B, T, C]
        x_global = self.global_transform(x_global).unsqueeze(2).expand(-1, -1, J, -1)  # [B, T, J, C]
        
        # 增强局部特征 - 为每个关节处理特征
        x_local = x.reshape(B * T * J, C)  # [B*T*J, C]
        x_local = self.local_enhancer(x_local).reshape(B, T, J, C)  # [B, T, J, C]
        x_local = self.local_transform(x_local)
        
        # 跨尺度注意力
        x_norm = self.norm1(x)
        x_flat = x_norm.reshape(B * T, J, C)
        global_flat = self.norm1(x_global).reshape(B * T, J, C)
        
        attn_output, _ = self.cross_scale_attn(
            query=x_flat,
            key=global_flat,
            value=global_flat
        )
        
        attn_output = attn_output.reshape(B, T, J, C)
        
        # 计算融合门控系数
        gate = self.fusion_gate(torch.cat([x_local, x_global], dim=-1))
        
        # 融合全局和局部特征
        x = x + gate * attn_output + (1 - gate) * x_local
        
        # 级联初始特征（如果提供）
        if initial_features is not None:
            x = x + initial_features * 0.1
        
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        
        return x

class BiomechanicalConstraintModule(nn.Module):
    def __init__(self, num_joints=17, dim_feat=128):
        super().__init__()
        
        # 关节角度预测器
        self.angle_predictor = nn.Sequential(
            nn.Linear(dim_feat, dim_feat // 2),
            nn.GELU(),
            nn.Linear(dim_feat // 2, 3)  # 欧拉角 (x, y, z)
        )
        
        # 定义人体骨架连接关系
        self.joint_connections = [
            (0, 1), (1, 2), (2, 3),  # 脊柱和头部
            (0, 4), (4, 5), (5, 6),  # 右臂
            (0, 7), (7, 8), (8, 9),  # 左臂
            (0, 10), (10, 11), (11, 12),  # 右腿
            (0, 13), (13, 14), (14, 15)   # 左腿
        ]
        
        # 关节角度限制 (min_angle, max_angle) in radians
        self.joint_angle_limits = {
            # 膝关节限制示例
            11: (0, 2.7),  # 右膝
            14: (0, 2.7),  # 左膝
            # 肘关节限制示例
            5: (0, 2.9),   # 右肘
            8: (0, 2.9),   # 左肘
            # 更多限制...
        }
        
        # 骨骼长度预测器
        self.bone_length_predictor = nn.Sequential(
            nn.Linear(dim_feat * 2, dim_feat),
            nn.GELU(),
            nn.Linear(dim_feat, 1),
            nn.ReLU()  # 保证长度为正
        )
        
        # 动作类型分类器，用于自适应权重
        self.action_classifier = nn.Sequential(
            nn.Linear(dim_feat * num_joints, 64),
            nn.GELU(),
            nn.Linear(64, 8)  # 8种常见动作类型
        )
    
    def compute_bone_lengths(self, pose3d):
        # pose3d: [B, T, J, 3]
        B, T, J, _ = pose3d.shape
        bone_lengths = []
        
        for joint_a, joint_b in self.joint_connections:
            # 计算骨骼长度
            vec = pose3d[:, :, joint_b, :] - pose3d[:, :, joint_a, :]
            length = torch.norm(vec, dim=-1, keepdim=True)  # [B, T, 1]
            bone_lengths.append(length)
            
        return torch.stack(bone_lengths, dim=2)  # [B, T, len(joint_connections), 1]
    
    def compute_joint_angles(self, pose3d):
        # pose3d: [B, T, J, 3]
        B, T, J, _ = pose3d.shape
        joint_angles = torch.zeros((B, T, J, 1), device=pose3d.device)
        
        for joint_id in self.joint_angle_limits.keys():
            # 找到与该关节相连的骨骼
            parent_connections = [(a, b) for a, b in self.joint_connections if b == joint_id]
            child_connections = [(a, b) for a, b in self.joint_connections if a == joint_id]
            
            if parent_connections and child_connections:
                parent_joint = parent_connections[0][0]
                child_joint = child_connections[0][1]
                
                # 计算两个骨骼向量
                v1 = pose3d[:, :, joint_id, :] - pose3d[:, :, parent_joint, :]
                v2 = pose3d[:, :, child_joint, :] - pose3d[:, :, joint_id, :]
                
                # 归一化向量
                v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-10)
                v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-10)
                
                # 计算角度 (arccos of dot product)
                dot_product = torch.sum(v1 * v2, dim=-1, keepdim=True)
                dot_product = torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7)
                angle = torch.acos(dot_product)
                
                joint_angles[:, :, joint_id, :] = angle
                
        return joint_angles
    
    def forward(self, features, pose3d):
        # features: [B, T, J, C]
        # pose3d: [B, T, J, 3]
        B, T, J, C = features.shape
        
        # 预测关节角度
        predicted_angles = self.angle_predictor(features)  # [B, T, J, 3]
        
        # 计算实际骨骼长度
        actual_bone_lengths = self.compute_bone_lengths(pose3d)  # [B, T, num_bones, 1]
        
        # 预测骨骼长度
        bone_features = []
        for joint_a, joint_b in self.joint_connections:
            # 连接相邻关节的特征
            joint_pair_feat = torch.cat([features[:, :, joint_a, :], features[:, :, joint_b, :]], dim=-1)
            bone_features.append(joint_pair_feat)
        
        bone_features = torch.stack(bone_features, dim=2)  # [B, T, num_bones, C*2]
        predicted_bone_lengths = self.bone_length_predictor(bone_features)  # [B, T, num_bones, 1]
        
        # 计算实际关节角度
        actual_angles = self.compute_joint_angles(pose3d)  # [B, T, J, 1]
        
        # 识别动作类型（用于自适应权重）
        action_features = features.reshape(B, T, -1).mean(dim=1)  # [B, J*C]
        action_logits = self.action_classifier(action_features)  # [B, 8]
        
        return predicted_angles, actual_angles, predicted_bone_lengths, actual_bone_lengths, action_logits

class ImprovedMemoryInducedTransformer(nn.Module):
    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, 
                 mlp_ratio=4, act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, 
                 graph_only=False, neighbour_num=4, n_frames=243,
                 use_hourglass_tokenizer=True, tokenizer_reduction_ratio=4, min_tokens=16,
                 use_global_local_comm=True):
        
        super().__init__()
        
        # 初始化中心姿态查询
        self.center_pose = nn.Parameter(torch.zeros(1, 1, num_joints, dim_feat))
        self.center_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, dim_feat))
        
        # 初始化关节嵌入和位置编码
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames, num_joints, dim_feat))
        
        # 初始化主干网络层
        self.layers = nn.ModuleList([
            TransformerBlock(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qkv_scale, drop=drop,
                           attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer,
                           use_layer_scale=use_layer_scale,
                           layer_scale_init_value=layer_scale_init_value)
            for _ in range(n_layers)
        ])
        
        # 初始化时间层
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qkv_scale, drop=drop,
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer,
                                   use_layer_scale=use_layer_scale,
                                   layer_scale_init_value=layer_scale_init_value,
                                   use_temporal_similarity=use_temporal_similarity,
                                   temporal_connection_len=temporal_connection_len,
                                   use_tcn=use_tcn, graph_only=graph_only,
                                   neighbour_num=neighbour_num)
            for _ in range(n_layers)
        ])
        
        # 初始化归一化层和输出层
        self.norm = nn.LayerNorm(dim_feat)
        self.rep_logit = nn.Linear(dim_feat, dim_rep)
        self.head = nn.Linear(dim_rep, dim_out)
        
        # 添加新的改进模块（如果启用）
        self.use_hourglass_tokenizer = use_hourglass_tokenizer
        if use_hourglass_tokenizer:
            self.hourglass_tokenizers = nn.ModuleList([
                HourglassTokenizer(dim=dim_feat, num_frames=n_frames, num_joints=num_joints,
                                 reduction_ratio=tokenizer_reduction_ratio, min_tokens=min_tokens)
                for _ in range(n_layers)
            ])
        
        self.use_global_local_comm = use_global_local_comm
        if use_global_local_comm:
            self.global_local_comms = nn.ModuleList([
                GlobalLocalCommunicationModule(dim=dim_feat, num_heads=num_heads,
                                             qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)
                for _ in range(n_layers)
            ])
    
    def get_features(self, x):
        """提取特征用于生物力学分析"""
        x = self.joints_embed(x)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return x
    
    def forward(self, x, return_rep=False):
        # x: [B, T, J, C]
        b, t, j, c = x.shape
        pose_query = self.center_pose.repeat(b, 1, 1, 1)  # [B, 1, J, C]
        pose_query = pose_query + self.center_pos_embed
        
        # 关节嵌入和位置编码
        x = self.joints_embed(x)
        x = x + self.pos_embed
        
        # 保存初始特征用于级联
        initial_features = x
        
        # 主干网络处理
        for i, (layer, temporal_layer) in enumerate(zip(self.layers, self.temporal_layers)):
            # 应用沙漏令牌化（如果启用）
            if self.use_hourglass_tokenizer:
                # 检测动作复杂度
                with torch.no_grad():
                    complexity = torch.std(x, dim=(1,2)).mean().item()
                    complexity_factor = min(1.0, max(0.3, complexity / 0.5))  # 归一化到0.3-1.0范围
                
                x = self.hourglass_tokenizers[i](x, complexity_factor=complexity_factor)
            
            # 常规处理
            x = layer(x)
            x, pose_query = temporal_layer(x, pose_query)
            
            # 全局局部通信（如果启用）
            if self.use_global_local_comm:
                x = self.global_local_comms[i](x, initial_features if i % 3 == 0 else None)
        
        # 后续处理不变
        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x
            
        x = self.head(x)
        return x

def biomechanical_loss(predicted_angles, actual_angles, predicted_bone_lengths, actual_bone_lengths, 
                       action_logits, target_actions, joint_angle_limits, motion_complexity):
    # 调整predicted_angles维度，只使用第一个通道进行损失计算
    # predicted_angles: [B, T, J, 3], actual_angles: [B, T, J, 1]
    predicted_angles_adjusted = predicted_angles[:, :, :, 0:1]  # 使用第一个通道 [B, T, J, 1]
    
    # 角度一致性损失
    angle_loss = F.mse_loss(predicted_angles_adjusted, actual_angles)
    
    # 骨骼长度一致性损失
    bone_length_loss = F.mse_loss(predicted_bone_lengths, actual_bone_lengths)
    
    # 骨骼长度变化约束 - 在时间维度上的方差应该很小
    bone_var_loss = torch.mean(torch.var(actual_bone_lengths, dim=1))
    
    # 关节角度范围约束
    angle_violation_loss = 0
    batch_size = predicted_angles.shape[0]
    
    for joint_id, (min_angle, max_angle) in joint_angle_limits.items():
        joint_angles = actual_angles[:, :, joint_id, 0]  # [B, T]
        below_min = torch.relu(min_angle - joint_angles)
        above_max = torch.relu(joint_angles - max_angle)
        angle_violation_loss += torch.mean(below_min + above_max)
    
    # 自适应权重基于动作类型
    if target_actions is not None:
        action_loss = F.cross_entropy(action_logits, target_actions)
        
        # 获取预测的动作置信度
        action_probs = F.softmax(action_logits, dim=1)
        action_confidence = torch.gather(action_probs, 1, target_actions.unsqueeze(1))
        
        # 根据动作类型和复杂度调整权重
        bone_weight = 1.0 + motion_complexity
        angle_weight = 1.0 + (1.0 - action_confidence.mean())
    else:
        action_loss = 0
        bone_weight = 1.0
        angle_weight = 1.0
    
    # 合并损失
    total_loss = angle_loss + bone_length_loss * bone_weight + bone_var_loss + angle_violation_loss * angle_weight
    
    return total_loss, {
        'angle_loss': angle_loss.item(),
        'bone_length_loss': bone_length_loss.item(),
        'bone_var_loss': bone_var_loss.item(),
        'angle_violation_loss': angle_violation_loss.item()
    }
