from torch import nn
import matplotlib.pyplot as plt
import torch
import torch.fft
import os


class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """
    
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial',vis = 'no'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.vis = vis
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)


class FrequencyAwareAttention(nn.Module):
    """
    Frequency-aware attention mechanism that processes both time and frequency domains
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='temporal', freq_ratio=0.5, use_low_freq=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.freq_ratio = freq_ratio  # Ratio of frequency domain processing
        self.use_low_freq = use_low_freq  # Whether to focus on low frequencies

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Frequency domain processing components
        self.freq_proj = nn.Linear(dim_in, dim_in)
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, T, J, C = x.shape

        # Standard attention computation
        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)

        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x_time = self.forward_temporal(q, k, v)
            x_freq = self.forward_frequency_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x_time = self.forward_spatial(q, k, v)
            x_freq = self.forward_frequency_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        # Adaptive fusion of time and frequency domain features
        gate = torch.sigmoid(self.freq_gate)
        x = gate * x_freq + (1 - gate) * x_time

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        """Standard spatial attention"""
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v):
        """Standard temporal attention"""
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_frequency_spatial(self, q, k, v):
        """Frequency-aware spatial attention"""
        B, H, T, J, C = q.shape

        # Apply FFT along temporal dimension for each joint
        q_freq = torch.fft.rfft(q, dim=2)  # (B, H, T//2+1, J, C)
        k_freq = torch.fft.rfft(k, dim=2)
        v_freq = torch.fft.rfft(v, dim=2)

        # Focus on low frequencies if specified
        if self.use_low_freq:
            freq_len = q_freq.shape[2]
            keep_len = max(1, int(freq_len * self.freq_ratio))
            q_freq = q_freq[:, :, :keep_len]
            k_freq = k_freq[:, :, :keep_len]
            v_freq = v_freq[:, :, :keep_len]

        # Compute attention in frequency domain
        attn_freq = (q_freq @ k_freq.transpose(-2, -1).conj()) * self.scale
        attn_freq = torch.softmax(attn_freq.real, dim=-1) + 1j * torch.softmax(attn_freq.imag, dim=-1)
        attn_freq = self.attn_drop(attn_freq)

        x_freq = attn_freq @ v_freq

        # Convert back to time domain
        if self.use_low_freq:
            # Pad with zeros for high frequencies
            freq_len_orig = T // 2 + 1
            x_freq_padded = torch.zeros(B, H, freq_len_orig, J, C, dtype=x_freq.dtype, device=x_freq.device)
            x_freq_padded[:, :, :x_freq.shape[2]] = x_freq
            x_freq = x_freq_padded

        x = torch.fft.irfft(x_freq, n=T, dim=2)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_frequency_temporal(self, q, k, v):
        """Frequency-aware temporal attention"""
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        # Apply FFT along temporal dimension
        qt_freq = torch.fft.rfft(qt, dim=3)  # (B, H, J, T//2+1, C)
        kt_freq = torch.fft.rfft(kt, dim=3)
        vt_freq = torch.fft.rfft(vt, dim=3)

        # Focus on low frequencies if specified
        if self.use_low_freq:
            freq_len = qt_freq.shape[3]
            keep_len = max(1, int(freq_len * self.freq_ratio))
            qt_freq = qt_freq[:, :, :, :keep_len]
            kt_freq = kt_freq[:, :, :, :keep_len]
            vt_freq = vt_freq[:, :, :, :keep_len]

        # Compute attention in frequency domain
        attn_freq = (qt_freq @ kt_freq.transpose(-2, -1).conj()) * self.scale
        attn_freq = torch.softmax(attn_freq.real, dim=-1) + 1j * torch.softmax(attn_freq.imag, dim=-1)
        attn_freq = self.attn_drop(attn_freq)

        x_freq = attn_freq @ vt_freq

        # Convert back to time domain
        if self.use_low_freq:
            # Pad with zeros for high frequencies
            freq_len_orig = T // 2 + 1
            x_freq_padded = torch.zeros(B, H, J, freq_len_orig, C, dtype=x_freq.dtype, device=x_freq.device)
            x_freq_padded[:, :, :, :x_freq.shape[3]] = x_freq
            x_freq = x_freq_padded

        x = torch.fft.irfft(x_freq, n=T, dim=3)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x
