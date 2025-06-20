from torch import nn
import torch
import torch.fft


class Sum_Attention(nn.Module):


    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,att_map,weight):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)

        q, k, v = qkv[0], qkv[1], qkv[2]
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        attn = weight*attn + (1-weight)*att_map

        attn = self.attn_drop(attn)
        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FrequencyAwareSumAttention(nn.Module):
    """
    Frequency-aware sum attention that combines self-attention with cross-attention maps
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial', freq_ratio=0.5, use_low_freq=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.freq_ratio = freq_ratio
        self.use_low_freq = use_low_freq

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Frequency domain components
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, x, att_map, weight):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)

        q, k, v = qkv[0], qkv[1], qkv[2]
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        # Time domain attention
        attn_time = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn_time = attn_time.softmax(dim=-1)
        attn_time = self.attn_drop(attn_time)

        # Frequency domain attention
        attn_freq = self.compute_frequency_attention(qt, kt, vt)

        # Adaptive fusion of time and frequency attention
        gate = torch.sigmoid(self.freq_gate)
        attn_self = gate * attn_freq + (1 - gate) * attn_time

        # Combine with cross-attention map
        attn = weight * attn_self + (1 - weight) * att_map

        attn = self.attn_drop(attn)
        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def compute_frequency_attention(self, qt, kt, vt):
        """Compute attention in frequency domain"""
        B, H, J, T, C = qt.shape

        # Apply FFT along temporal dimension
        qt_freq = torch.fft.rfft(qt, dim=3)  # (B, H, J, T//2+1, C)
        kt_freq = torch.fft.rfft(kt, dim=3)

        # Focus on low frequencies if specified
        if self.use_low_freq:
            freq_len = qt_freq.shape[3]
            keep_len = max(1, int(freq_len * self.freq_ratio))
            qt_freq = qt_freq[:, :, :, :keep_len]
            kt_freq = kt_freq[:, :, :, :keep_len]

        # Compute attention in frequency domain
        attn_freq = (qt_freq @ kt_freq.transpose(-2, -1).conj()) * self.scale
        attn_freq = torch.softmax(attn_freq.real, dim=-1) + 1j * torch.softmax(attn_freq.imag, dim=-1)

        # Convert back to time domain for consistency
        if self.use_low_freq:
            # Pad with zeros for high frequencies
            freq_len_orig = T // 2 + 1
            attn_freq_padded = torch.zeros(B, H, J, freq_len_orig, attn_freq.shape[4],
                                         dtype=attn_freq.dtype, device=attn_freq.device)
            attn_freq_padded[:, :, :, :attn_freq.shape[3], :attn_freq.shape[4]] = attn_freq
            attn_freq = attn_freq_padded

        # For attention map, we need to interpolate back to original temporal size
        if attn_freq.shape[3] != T or attn_freq.shape[4] != T:
            # Use the real part and interpolate to match expected dimensions
            attn_freq_real = torch.nn.functional.interpolate(
                attn_freq.real.view(B*H*J, attn_freq.shape[3], attn_freq.shape[4]).unsqueeze(1),
                size=(T, T),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).view(B, H, J, T, T)
        else:
            attn_freq_real = attn_freq.real

        return attn_freq_real







