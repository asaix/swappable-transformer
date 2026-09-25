import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    def __init__(self, d_emb, msl=5000): # msl = min. seq. len
        super().__init__()

        # follows AIAYN formula, which is defined only for even d_emb
        assert d_emb % 2 == 0, "implementation requires even-dimensional embedding"

        # each row (↓ headers) = position in sequence
        # each col (-> headers) = a dimension of the embedding
        # so pemb[row X][col Y] = PE for dimension Y when token at position X
        pemb = torch.zeros(msl, d_emb)

        positions = torch.arange(0, msl).unsqueeze(1)
        denominators = torch.pow(
            10000,
            torch.arange(0, d_emb, 2) / d_emb
        ) # = 10K ^ (2(i=1, 2, 3...d_emb) / d_emb)
        angles = positions / denominators

        pemb[:, 0::2] = torch.sin(angles)
        pemb[:, 1::2] = torch.cos(angles)
        self.register_buffer("pemb", pemb)
        
    def forward(self, x):
        return x + self.pemb[:x.size(1), :]


class RoPE(nn.Module):
    def __init__(self, d_emb): 
        super().__init__()
        self.d_emb = d_emb

        assert self.d_emb % 2 == 0, "implementation requires even-dimensional embedding"

        # from the RoPE paper => 10000^(-2i/d_emb)
        theta = 1 / (10000 ** (torch.arange(0, self.d_emb, 2) / self.d_emb))
        theta = theta.repeat_interleave(2) # [1, 1, 2, 2, ...]

        self.register_buffer("theta", theta)

    def forward(self, x):
        B, H, S, D = x.shape

        positions = torch.arange(S, device=x.device)
        angles = torch.outer(positions, self.theta)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # make (S, D) -> (1, 1, S, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # x1, x2, ... => -x2, x1, -x4, x3 ... -xd-1 xd
        x_rotated = torch.stack(
            (-x[..., 1::2], x[..., ::2]),
            dim=-1
        ).flatten(-2)
        
        return x * cos + x_rotated * sin
    
    
        