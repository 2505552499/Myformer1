import torch
import torch.nn as nn
import torch.nn.functional as F


class JointFlowV2(nn.Module):
    """
    JointFlow V2: Improved motion encoding module
    
    Improvements:
    1. Better initialization strategy
    2. More aggressive motion scaling
    3. Learnable motion computation
    4. Multi-scale temporal modeling
    """
    
    def __init__(self, dim_in=3, motion_dim=32, use_velocity=True, use_acceleration=True, 
                 dropout=0.1, aggressive_scaling=True):
        super(JointFlowV2, self).__init__()
        
        self.dim_in = dim_in
        self.motion_dim = motion_dim
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.aggressive_scaling = aggressive_scaling
        
        # Calculate feature dimensions
        feature_dims = []
        if use_velocity:
            feature_dims.append(motion_dim // 2)
        if use_acceleration:
            feature_dims.append(motion_dim // 2)
        
        if not feature_dims:
            raise ValueError("At least one of velocity or acceleration must be enabled")
        
        if len(feature_dims) == 1:
            feature_dims[0] = motion_dim
        
        # Learnable motion computation weights
        if self.use_velocity:
            self.velocity_weights = nn.Parameter(torch.tensor([1.0, -1.0]))  # [t, t-1]
            
        if self.use_acceleration:
            self.acceleration_weights = nn.Parameter(torch.tensor([1.0, -2.0, 1.0]))  # [t, t-1, t-2]
        
        # Motion feature encoders with better architecture
        if self.use_velocity:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(dim_in, feature_dims[0] * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dims[0] * 2, feature_dims[0]),
                nn.ReLU(inplace=True)
            )
        
        if self.use_acceleration:
            acc_idx = 1 if self.use_velocity else 0
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(dim_in, feature_dims[acc_idx] * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dims[acc_idx] * 2, feature_dims[acc_idx]),
                nn.ReLU(inplace=True)
            )
        
        # Improved motion fusion with residual connection
        self.motion_fusion = nn.Sequential(
            nn.Linear(motion_dim, motion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(motion_dim, dim_in)
            # Remove Tanh to allow larger enhancements
        )
        
        # More aggressive learnable scaling
        if aggressive_scaling:
            self.motion_scale = nn.Parameter(torch.tensor(1.0))  # Start with 1.0 instead of 0.1
        else:
            self.motion_scale = nn.Parameter(torch.tensor(0.1))
        
        # Multi-scale temporal weights
        self.temporal_weights = nn.Parameter(torch.ones(3))  # For different time scales
        
        self._init_weights()
    
    def _init_weights(self):
        """Improved initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use normal Xavier initialization instead of small gain
                nn.init.xavier_uniform_(m.weight, gain=1.0)  # Changed from 0.1 to 1.0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_velocity_learnable(self, x):
        """Compute velocity with learnable weights"""
        B, T, J, C = x.shape
        
        if T <= 1:
            return torch.zeros_like(x)
        
        # Learnable velocity computation
        velocity = torch.zeros_like(x)
        
        # Apply learnable weights: w1*x[t] + w2*x[t-1]
        velocity[:, 1:] = (self.velocity_weights[0] * x[:, 1:] + 
                          self.velocity_weights[1] * x[:, :-1])
        
        # For the first frame, use the velocity of the second frame
        velocity[:, 0] = velocity[:, 1]
        
        return velocity
    
    def compute_acceleration_learnable(self, x):
        """Compute acceleration with learnable weights"""
        B, T, J, C = x.shape
        
        if T <= 2:
            return torch.zeros_like(x)
        
        # Learnable acceleration computation: w1*x[t] + w2*x[t-1] + w3*x[t-2]
        acceleration = torch.zeros_like(x)
        acceleration[:, 2:] = (self.acceleration_weights[0] * x[:, 2:] + 
                              self.acceleration_weights[1] * x[:, 1:-1] + 
                              self.acceleration_weights[2] * x[:, :-2])
        
        # Handle boundary conditions
        acceleration[:, 0] = acceleration[:, 2] if T > 2 else torch.zeros_like(acceleration[:, 0])
        acceleration[:, 1] = acceleration[:, 2] if T > 2 else torch.zeros_like(acceleration[:, 1])
        
        return acceleration
    
    def compute_multi_scale_features(self, x):
        """Compute multi-scale temporal features"""
        B, T, J, C = x.shape
        
        # Different time scales
        scales = [1, 2, 4]  # 1-frame, 2-frame, 4-frame differences
        multi_scale_features = []
        
        for i, scale in enumerate(scales):
            if T > scale:
                diff = x[:, scale:] - x[:, :-scale]
                # Pad to maintain temporal dimension
                padded_diff = F.pad(diff, (0, 0, 0, 0, scale, 0), mode='replicate')
                multi_scale_features.append(self.temporal_weights[i] * padded_diff)
            else:
                multi_scale_features.append(torch.zeros_like(x))
        
        # Combine multi-scale features
        combined = sum(multi_scale_features) / len(multi_scale_features)
        return combined
    
    def forward(self, x):
        """
        Forward pass with improved motion modeling
        """
        B, T, J, C = x.shape
        
        motion_features = []
        
        # Compute and encode velocity features
        if self.use_velocity:
            velocity = self.compute_velocity_learnable(x)
            vel_feat = self.velocity_encoder(velocity)
            motion_features.append(vel_feat)
        
        # Compute and encode acceleration features
        if self.use_acceleration:
            acceleration = self.compute_acceleration_learnable(x)
            acc_feat = self.acceleration_encoder(acceleration)
            motion_features.append(acc_feat)
        
        # Concatenate motion features
        motion_feat = torch.cat(motion_features, dim=-1)  # [B, T, J, motion_dim]
        
        # Fuse motion features
        motion_enhancement = self.motion_fusion(motion_feat)  # [B, T, J, C]
        
        # Add multi-scale temporal features
        multi_scale_enhancement = self.compute_multi_scale_features(x)
        total_enhancement = motion_enhancement + 0.1 * multi_scale_enhancement
        
        # Apply learnable scaling with residual connection
        enhanced_x = x + self.motion_scale * total_enhancement
        
        return enhanced_x


def test_joint_flow_v2():
    """Test the improved JointFlow V2"""
    print("Testing JointFlow V2...")
    
    B, T, J, C = 2, 81, 17, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input with clear motion pattern
    x = torch.zeros(B, T, J, C).to(device)
    for t in range(T):
        phase = 2 * torch.pi * t / T
        x[:, t, :, 0] = torch.sin(phase) * 0.1
        x[:, t, :, 1] = torch.cos(phase * 2) * 0.05
    
    # Test V1 vs V2
    joint_flow_v1 = JointFlow(dim_in=C, motion_dim=32).to(device)
    joint_flow_v2 = JointFlowV2(dim_in=C, motion_dim=32, aggressive_scaling=True).to(device)
    
    with torch.no_grad():
        enhanced_v1 = joint_flow_v1(x)
        enhanced_v2 = joint_flow_v2(x)
    
    # Compare enhancement magnitudes
    enhancement_v1 = torch.mean(torch.abs(enhanced_v1 - x)).item()
    enhancement_v2 = torch.mean(torch.abs(enhanced_v2 - x)).item()
    
    print(f"V1 enhancement magnitude: {enhancement_v1:.6f}")
    print(f"V2 enhancement magnitude: {enhancement_v2:.6f}")
    print(f"V2 improvement: {enhancement_v2/enhancement_v1:.2f}x")
    
    # Check learnable parameters
    print(f"\nV2 velocity weights: {joint_flow_v2.velocity_weights.data}")
    print(f"V2 acceleration weights: {joint_flow_v2.acceleration_weights.data}")
    print(f"V2 motion scale: {joint_flow_v2.motion_scale.data}")
    print(f"V2 temporal weights: {joint_flow_v2.temporal_weights.data}")


if __name__ == "__main__":
    # Import V1 for comparison
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.modules.joint_flow import JointFlow
    
    test_joint_flow_v2()
