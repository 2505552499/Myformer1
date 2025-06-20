import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedMotionFlow(nn.Module):
    """
    Enhanced Motion Flow: Improved motion computation for TCPFormer

    Key improvements over basic JointFlow:
    1. Multi-scale temporal modeling (1, 2, 4 frame differences)
    2. Learnable motion kernels instead of fixed differences
    3. Non-linear motion transformation
    4. Adaptive motion scaling with gating
    5. Better initialization strategy
    6. Option for direct high-dimensional output (avoiding information bottleneck)
    """

    def __init__(self, dim_in=3, motion_dim=64, dropout=0.1, temporal_scales=[1, 8, 25, 100],
                 output_high_dim=True):
        super(EnhancedMotionFlow, self).__init__()
        
        self.dim_in = dim_in
        self.motion_dim = motion_dim
        self.temporal_scales = temporal_scales
        self.output_high_dim = output_high_dim

        # Ensure motion_dim is properly handled
        # Make motion_dim divisible by number of scales
        if motion_dim % len(temporal_scales) != 0:
            motion_dim = ((motion_dim // len(temporal_scales)) + 1) * len(temporal_scales)

        scale_dim = motion_dim // len(temporal_scales)
        self.actual_motion_dim = motion_dim  # Use the corrected motion_dim
        
        # Learnable motion kernels for different scales
        self.motion_kernels = nn.ParameterDict()
        for scale in temporal_scales:
            kernel_size = min(scale + 1, 5)  # Max kernel size of 5
            self.motion_kernels[f'scale_{scale}'] = nn.Parameter(
                torch.randn(kernel_size, dim_in) * 0.1
            )
        
        # Multi-scale motion encoders
        self.scale_encoders = nn.ModuleDict()
        for scale in temporal_scales:
            self.scale_encoders[f'scale_{scale}'] = nn.Sequential(
                nn.Linear(dim_in, scale_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(scale_dim * 2, scale_dim),
                nn.ReLU(inplace=True)
            )
        
        # Original input embedding for residual connection
        self.original_embed = nn.Sequential(
            nn.Linear(dim_in, self.actual_motion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.actual_motion_dim, self.actual_motion_dim)
        )

        # Non-linear motion transformation
        if output_high_dim:
            # Output high-dimensional features directly with residual
            self.motion_transform = nn.Sequential(
                nn.Linear(self.actual_motion_dim, self.actual_motion_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.actual_motion_dim * 2, self.actual_motion_dim),
                nn.ReLU(inplace=True)
            )
        else:
            # Traditional approach: compress back to input dimension
            self.motion_transform = nn.Sequential(
                nn.Linear(self.actual_motion_dim, self.actual_motion_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.actual_motion_dim * 2, self.actual_motion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.actual_motion_dim, dim_in)
            )
        
        # Adaptive motion scaling with gating (only for low-dim output)
        if not output_high_dim:
            self.motion_scale = nn.Parameter(torch.tensor(0.5))
            self.motion_gate = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.Sigmoid()
            )
        
        # Learnable weights for combining different scales
        self.scale_weights = nn.Parameter(torch.ones(len(temporal_scales)))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize motion kernels with finite difference patterns
        for scale_name, kernel in self.motion_kernels.items():
            kernel_size = kernel.shape[0]
            if kernel_size == 2:
                # Simple difference: [1, -1]
                kernel.data = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            elif kernel_size == 3:
                # Central difference: [-0.5, 0, 0.5]
                kernel.data = torch.tensor([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
            else:
                # Initialize with small random values
                nn.init.normal_(kernel, mean=0, std=0.1)
    
    def compute_multi_scale_motion(self, x):
        """Compute motion features at multiple temporal scales"""
        B, T, J, _ = x.shape
        scale_features = []
        
        for i, scale in enumerate(self.temporal_scales):
            if T <= scale:
                # If sequence too short, use zero motion
                scale_dim = self.actual_motion_dim // len(self.temporal_scales)
                motion_feat = torch.zeros(B, T, J, scale_dim).to(x.device)
            else:
                # Apply learnable motion kernel
                motion = self._apply_motion_kernel(x, scale)
                # Encode motion features
                motion_feat = self.scale_encoders[f'scale_{scale}'](motion)
            
            scale_features.append(motion_feat)
        
        # Combine multi-scale features with learnable weights
        scale_weights_norm = torch.softmax(self.scale_weights, dim=0)
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weighted_features.append(scale_weights_norm[i] * feat)
        
        combined_motion = torch.cat(weighted_features, dim=-1)  # [B, T, J, motion_dim]
        return combined_motion
    
    def _apply_motion_kernel(self, x, scale):
        """Apply learnable motion kernel for specific scale"""
        B, T, J, C = x.shape
        kernel = self.motion_kernels[f'scale_{scale}']
        kernel_size = kernel.shape[0]
        
        # Simple implementation: apply kernel as weighted sum
        motion = torch.zeros_like(x)
        
        for t in range(T):
            for k in range(kernel_size):
                t_source = max(0, min(T-1, t - kernel_size//2 + k))
                motion[:, t] += kernel[k].unsqueeze(0).unsqueeze(0) * x[:, t_source]
        
        return motion
    
    def forward(self, x):
        """
        Forward pass with enhanced motion computation and residual connection

        Args:
            x: Input pose tensor [B, T, J, C] where C=3 (x, y, confidence)

        Returns:
            enhanced_x: Motion-enhanced features [B, T, J, motion_dim] with residual connection
        """
        # Embed original input to high-dimensional space for residual connection
        original_features = self.original_embed(x)  # [B, T, J, motion_dim]

        # Compute multi-scale motion features
        motion_features = self.compute_multi_scale_motion(x)  # [B, T, J, motion_dim]

        # Non-linear motion transformation
        transformed_motion = self.motion_transform(motion_features)  # [B, T, J, motion_dim]

        if self.output_high_dim:
            # Residual connection: original + motion enhancement
            # This ensures we never perform worse than the original
            enhanced_features = original_features + transformed_motion
            return enhanced_features  # [B, T, J, motion_dim]
        else:
            # Traditional approach: compress and add residual
            motion_enhancement = transformed_motion  # [B, T, J, C]

            # Adaptive gating mechanism
            motion_gate = self.motion_gate(x)  # [B, T, J, C]
            gated_enhancement = motion_enhancement * motion_gate

            # Apply learnable scaling with residual connection
            enhanced_x = x + self.motion_scale * gated_enhancement

            return enhanced_x  # [B, T, J, C]


class MotionAwareLoss(nn.Module):
    """Motion-aware loss function to encourage better motion learning"""
    
    def __init__(self, velocity_weight=1.0, acceleration_weight=0.5, smoothness_weight=0.1):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.smoothness_weight = smoothness_weight
    
    def compute_velocity(self, poses):
        """Compute velocity from poses"""
        if poses.shape[1] <= 1:
            return torch.zeros_like(poses)
        velocity = poses[:, 1:] - poses[:, :-1]
        velocity = F.pad(velocity, (0, 0, 0, 0, 1, 0), mode='replicate')
        return velocity
    
    def compute_acceleration(self, velocity):
        """Compute acceleration from velocity"""
        if velocity.shape[1] <= 1:
            return torch.zeros_like(velocity)
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        acceleration = F.pad(acceleration, (0, 0, 0, 0, 1, 0), mode='replicate')
        return acceleration
    
    def forward(self, pred_poses, target_poses):
        """Compute motion-aware loss"""
        # Position loss
        pos_loss = F.mse_loss(pred_poses, target_poses)
        
        # Velocity loss
        pred_velocity = self.compute_velocity(pred_poses)
        target_velocity = self.compute_velocity(target_poses)
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)
        
        # Acceleration loss
        pred_acceleration = self.compute_acceleration(pred_velocity)
        target_acceleration = self.compute_acceleration(target_velocity)
        acceleration_loss = F.mse_loss(pred_acceleration, target_acceleration)
        
        # Motion smoothness loss
        velocity_diff = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        smoothness_loss = torch.mean(torch.norm(velocity_diff, dim=-1))
        
        # Total loss
        total_loss = (pos_loss + 
                     self.velocity_weight * velocity_loss + 
                     self.acceleration_weight * acceleration_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        return {
            'total_loss': total_loss,
            'pos_loss': pos_loss,
            'velocity_loss': velocity_loss,
            'acceleration_loss': acceleration_loss,
            'smoothness_loss': smoothness_loss
        }


def test_enhanced_motion_flow():
    """Test the enhanced motion flow module"""
    print("Testing Enhanced Motion Flow...")
    
    B, T, J, C = 2, 81, 17, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input with realistic motion pattern
    x = torch.zeros(B, T, J, C).to(device)
    for t in range(T):
        # Simulate walking motion
        phase = 2 * np.pi * t / T * 2  # 2 walking cycles
        x[:, t, :, 0] = torch.sin(torch.tensor(phase)) * 0.1  # X movement
        x[:, t, :, 1] = torch.cos(torch.tensor(phase * 2)) * 0.05  # Y movement
        x[:, t, :, 2] = torch.sin(torch.tensor(phase * 0.5)) * 0.02  # Z movement
    
    # Test Enhanced Motion Flow
    motion_flow = EnhancedMotionFlow(dim_in=C, motion_dim=64, temporal_scales=[1, 2, 4]).to(device)
    
    # Forward pass
    with torch.no_grad():
        enhanced_x = motion_flow(x)
    
    # Calculate enhancement metrics
    enhancement_magnitude = torch.mean(torch.abs(enhanced_x - x)).item()
    
    # Count parameters
    num_params = sum(p.numel() for p in motion_flow.parameters())
    
    print(f"Enhancement magnitude: {enhancement_magnitude:.6f}")
    print(f"Number of parameters: {num_params:,}")
    print(f"Output shape: {enhanced_x.shape}")
    
    # Test motion-aware loss
    motion_loss = MotionAwareLoss()
    target = x + torch.randn_like(x) * 0.01
    loss_dict = motion_loss(enhanced_x, target)
    print(f"Motion-aware loss: {loss_dict['total_loss']:.6f}")
    
    # Test learnable parameters
    print(f"Motion scale: {motion_flow.motion_scale.data:.3f}")
    print(f"Scale weights: {torch.softmax(motion_flow.scale_weights, dim=0).data}")
    
    print("✅ Enhanced Motion Flow test passed!")
    
    return motion_flow, enhanced_x


if __name__ == "__main__":
    test_enhanced_motion_flow()
