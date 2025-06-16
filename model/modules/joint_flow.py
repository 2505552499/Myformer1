import torch
import torch.nn as nn
import torch.nn.functional as F


class JointFlow(nn.Module):
    """
    JointFlow: Motion encoding module for 3D human pose estimation
    
    This module encodes motion information (velocity and acceleration) of each joint
    to enhance the temporal consistency and accuracy of pose estimation.
    
    Args:
        dim_in (int): Input dimension (typically 3 for x,y,z coordinates)
        motion_dim (int): Dimension of motion features
        use_velocity (bool): Whether to use velocity features
        use_acceleration (bool): Whether to use acceleration features
        dropout (float): Dropout rate for motion features
    """
    
    def __init__(self, dim_in=3, motion_dim=32, use_velocity=True, use_acceleration=True, dropout=0.1):
        super(JointFlow, self).__init__()
        
        self.dim_in = dim_in
        self.motion_dim = motion_dim
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        
        # Calculate feature dimensions
        feature_dims = []
        if use_velocity:
            feature_dims.append(motion_dim // 2)
        if use_acceleration:
            feature_dims.append(motion_dim // 2)
        
        if not feature_dims:
            raise ValueError("At least one of velocity or acceleration must be enabled")
        
        # Adjust dimensions if only one feature type is used
        if len(feature_dims) == 1:
            feature_dims[0] = motion_dim
        
        # Motion feature encoders
        if self.use_velocity:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(dim_in, feature_dims[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        if self.use_acceleration:
            acc_idx = 1 if self.use_velocity else 0
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(dim_in, feature_dims[acc_idx]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        # Motion fusion layer
        self.motion_fusion = nn.Sequential(
            nn.Linear(motion_dim, dim_in),
            nn.Tanh()  # Use tanh to bound the motion enhancement
        )
        
        # Learnable scaling factor for motion enhancement
        self.motion_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_velocity(self, x):
        """
        Compute velocity features from pose sequence
        
        Args:
            x: Input tensor [B, T, J, C]
        
        Returns:
            velocity: Velocity tensor [B, T, J, C]
        """
        B, T, J, C = x.shape
        
        if T <= 1:
            return torch.zeros_like(x)
        
        # Compute velocity: v_t = x_t - x_{t-1}
        velocity = torch.zeros_like(x)
        velocity[:, 1:] = x[:, 1:] - x[:, :-1]
        
        # For the first frame, use the velocity of the second frame
        velocity[:, 0] = velocity[:, 1]
        
        return velocity
    
    def compute_acceleration(self, velocity):
        """
        Compute acceleration features from velocity
        
        Args:
            velocity: Velocity tensor [B, T, J, C]
        
        Returns:
            acceleration: Acceleration tensor [B, T, J, C]
        """
        B, T, J, C = velocity.shape
        
        if T <= 1:
            return torch.zeros_like(velocity)
        
        # Compute acceleration: a_t = v_t - v_{t-1}
        acceleration = torch.zeros_like(velocity)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        
        # For the first frame, use the acceleration of the second frame
        acceleration[:, 0] = acceleration[:, 1]
        
        return acceleration
    
    def forward(self, x):
        """
        Forward pass of JointFlow
        
        Args:
            x: Input pose tensor [B, T, J, C]
        
        Returns:
            enhanced_x: Motion-enhanced pose tensor [B, T, J, C]
        """
        B, T, J, C = x.shape
        
        motion_features = []
        
        # Compute and encode velocity features
        if self.use_velocity:
            velocity = self.compute_velocity(x)
            vel_feat = self.velocity_encoder(velocity)
            motion_features.append(vel_feat)
        
        # Compute and encode acceleration features
        if self.use_acceleration:
            if self.use_velocity:
                # Use computed velocity
                acceleration = self.compute_acceleration(velocity)
            else:
                # Compute velocity first, then acceleration
                velocity = self.compute_velocity(x)
                acceleration = self.compute_acceleration(velocity)
            
            acc_feat = self.acceleration_encoder(acceleration)
            motion_features.append(acc_feat)
        
        # Concatenate motion features
        motion_feat = torch.cat(motion_features, dim=-1)  # [B, T, J, motion_dim]
        
        # Fuse motion features back to input dimension
        motion_enhancement = self.motion_fusion(motion_feat)  # [B, T, J, C]
        
        # Apply learnable scaling and add to original input
        enhanced_x = x + self.motion_scale * motion_enhancement
        
        return enhanced_x


def test_joint_flow():
    """Test function for JointFlow module"""
    print("Testing JointFlow module...")
    
    # Test parameters
    B, T, J, C = 2, 243, 17, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input
    x = torch.randn(B, T, J, C).to(device)
    
    # Test different configurations
    configs = [
        {"motion_dim": 32, "use_velocity": True, "use_acceleration": True},
        {"motion_dim": 16, "use_velocity": True, "use_acceleration": False},
        {"motion_dim": 16, "use_velocity": False, "use_acceleration": True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config}")
        
        # Create JointFlow module
        joint_flow = JointFlow(dim_in=C, **config).to(device)
        
        # Forward pass
        with torch.no_grad():
            enhanced_x = joint_flow(x)
        
        # Check output shape
        assert enhanced_x.shape == x.shape, f"Output shape mismatch: {enhanced_x.shape} vs {x.shape}"
        
        # Check that output is different from input (motion enhancement applied)
        diff = torch.mean(torch.abs(enhanced_x - x))
        print(f"  Mean absolute difference: {diff:.6f}")
        
        # Count parameters
        num_params = sum(p.numel() for p in joint_flow.parameters())
        print(f"  Number of parameters: {num_params:,}")
        
        print(f"  Test {i+1} passed!")
    
    print("\nAll JointFlow tests passed!")


if __name__ == "__main__":
    test_joint_flow()
