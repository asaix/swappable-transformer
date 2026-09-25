import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_emb, eps=1e-5):
        super().__init__()

        # later gamma * x_norm + beta so init gamma w/ ones and beta w/ zeroes
        self.gamma = nn.Parameter(torch.ones(d_emb))
        self.beta = nn.Parameter(torch.zeros(d_emb))

        self.eps = eps 
        # x_norm = (x - mean) / √(variance + ε)
        # if var = 0 then use 0.00001 so we don't div by zero
        
    def forward(self, x):
        # x = (Batch, Seqlen, Embedding_vals)
        mean = x.mean(dim=-1, keepdim=True) 
        var = x.var(dim=-1, keepdim=True, unbiased=False) # use population variance (divide by N not N-1)
        
        x_norm = (x-mean) / (var + self.eps).sqrt()
        
        return self.gamma * x_norm + self.beta
    
    
class RMSNorm(nn.Module):
    # RMSNorm(x) =[ x / sqrt(mean(x^2) + eps) ] * gamma
    def __init__(self, d_emb, eps=1e-5):
        super().__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_emb))
        
    def forward(self, x):
        x_rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.gamma * x / x_rms
    
    