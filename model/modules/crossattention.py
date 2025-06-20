import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import os



class CrossAttention(nn.Module):

    def __init__(self,dim_in,dim_out,num_heads = 8,qkv_bias = False,qkv_scale = None,attn_drop=0.,proj_drop=0.,
                 mode = 'temporal',back_att = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qkv_scale or head_dim**(-0.5)
        self.wq = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.wk = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.wv = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in,dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.back_att = back_att

    def forward(self,q,kv):
        #batch_size temporal_frame_num num_joints feature_dim
        b , t , j , d = q.shape
        t_sup = kv.shape[1]
        q = self.wq(q).reshape(b,t,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)
        k = self.wk(kv).reshape(b,t_sup,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)
        v = self.wv(kv).reshape(b,t_sup,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)

        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn@v  # b h j t c
        out = out.permute(0,3,2,1,4).reshape(b,t,j,d)
        out = self.proj(out)
        out = self.proj_drop(out)
        if self.back_att:
            return attn,out
        else:
            return out



class FrequencyAwareCrossAttention(nn.Module):
    """
    Frequency-aware cross attention that processes both time and frequency domains
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0., proj_drop=0.,
                 mode='temporal', back_att=None, freq_ratio=0.5, use_low_freq=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qkv_scale or head_dim**(-0.5)
        self.freq_ratio = freq_ratio
        self.use_low_freq = use_low_freq

        self.wq = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.wk = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.wv = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.back_att = back_att

        # Frequency domain components
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, q, kv):
        b, t, j, d = q.shape
        t_sup = kv.shape[1]

        q_proj = self.wq(q).reshape(b, t, j, self.num_heads, d//self.num_heads).permute(0, 3, 2, 1, 4)
        k_proj = self.wk(kv).reshape(b, t_sup, j, self.num_heads, d//self.num_heads).permute(0, 3, 2, 1, 4)
        v_proj = self.wv(kv).reshape(b, t_sup, j, self.num_heads, d//self.num_heads).permute(0, 3, 2, 1, 4)

        # Time domain attention
        attn_time = (q_proj @ k_proj.transpose(-2, -1)) * self.scale
        attn_time = attn_time.softmax(dim=-1)
        attn_time = self.attn_drop(attn_time)
        out_time = attn_time @ v_proj

        # Frequency domain attention
        out_freq, attn_freq = self.forward_frequency_domain(q_proj, k_proj, v_proj)

        # Adaptive fusion
        gate = torch.sigmoid(self.freq_gate)
        attn = gate * attn_freq + (1 - gate) * attn_time
        out = gate * out_freq + (1 - gate) * out_time

        out = out.permute(0, 3, 2, 1, 4).reshape(b, t, j, d)
        out = self.proj(out)
        out = self.proj_drop(out)

        if self.back_att:
            return attn, out
        else:
            return out

    def forward_frequency_domain(self, q, k, v):
        """Process attention in frequency domain"""
        B, H, J, T_q, C = q.shape
        T_kv = k.shape[3]

        # Apply FFT along temporal dimension
        q_freq = torch.fft.rfft(q, dim=3)  # (B, H, J, T_q//2+1, C)
        k_freq = torch.fft.rfft(k, dim=3)  # (B, H, J, T_kv//2+1, C)
        v_freq = torch.fft.rfft(v, dim=3)  # (B, H, J, T_kv//2+1, C)

        # Focus on low frequencies if specified
        if self.use_low_freq:
            freq_len_q = q_freq.shape[3]
            freq_len_kv = k_freq.shape[3]
            keep_len_q = max(1, int(freq_len_q * self.freq_ratio))
            keep_len_kv = max(1, int(freq_len_kv * self.freq_ratio))

            q_freq = q_freq[:, :, :, :keep_len_q]
            k_freq = k_freq[:, :, :, :keep_len_kv]
            v_freq = v_freq[:, :, :, :keep_len_kv]

        # Compute attention in frequency domain
        attn_freq = (q_freq @ k_freq.transpose(-2, -1).conj()) * self.scale
        attn_freq = torch.softmax(attn_freq.real, dim=-1) + 1j * torch.softmax(attn_freq.imag, dim=-1)
        attn_freq = self.attn_drop(attn_freq)

        out_freq = attn_freq @ v_freq

        # Convert back to time domain
        if self.use_low_freq:
            # Pad with zeros for high frequencies
            freq_len_orig = T_q // 2 + 1
            out_freq_padded = torch.zeros(B, H, J, freq_len_orig, C, dtype=out_freq.dtype, device=out_freq.device)
            out_freq_padded[:, :, :, :out_freq.shape[3]] = out_freq
            out_freq = out_freq_padded

            # Also pad attention map for consistency
            attn_freq_padded = torch.zeros(B, H, J, freq_len_orig, k_freq.shape[3], dtype=attn_freq.dtype, device=attn_freq.device)
            attn_freq_padded[:, :, :, :attn_freq.shape[3], :attn_freq.shape[4]] = attn_freq
            attn_freq = attn_freq_padded

        out_freq = torch.fft.irfft(out_freq, n=T_q, dim=3)

        # For attention map, we need to interpolate back to original temporal size
        if attn_freq.shape[3] != T_q or attn_freq.shape[4] != k_freq.shape[3]:
            # Use the real part and interpolate to match expected dimensions
            attn_freq_real = torch.nn.functional.interpolate(
                attn_freq.real.view(B*H*J, attn_freq.shape[3], attn_freq.shape[4]).unsqueeze(1),
                size=(T_q, T_kv),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).view(B, H, J, T_q, T_kv)
        else:
            attn_freq_real = attn_freq.real

        return out_freq, attn_freq_real


# x = torch.ones((16,9,17,256))
# kv = torch.ones((16,18,17,256))

# net = CrossAttention(dim_in=256,dim_out=256,head_num=8)

# out = net(x,kv)
