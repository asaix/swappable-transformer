import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, d_emb=512, num_heads=8, rope=None):
        super().__init__()

        assert d_emb % num_heads == 0

        self.d_emb = d_emb
        self.num_heads = num_heads
        self.head_dim = d_emb // num_heads
        self.rope = rope # RoPE(head_dim) for C2, None otherwise

        self.W_q = nn.Linear(d_emb, d_emb)
        self.W_k = nn.Linear(d_emb, d_emb)
        self.W_v = nn.Linear(d_emb, d_emb)

        self.W_o = nn.Linear(d_emb, d_emb)

    def forward(self, x, mask=None, encoder_output=None):

        # Q, K, V => (n x head_dim)
        Q = self.W_q(x)

        if encoder_output is not None:
            # decoder cross attention
            K = self.W_k(encoder_output)
            V = self.W_v(encoder_output)
        else:
            # normal encoder attention
            K = self.W_k(x)
            V = self.W_v(x)

        # Mult-Head bit
        # B, S, D = Batch size, Seq length (n), head_dim
        # Instead of making multiple heads, we change the way the matrix is interpreted ie [a b c d] => [a b], [c d]
        # Practically, there's only one Wq Wk and Wv but by transposing into (B, H, S, head_dim) each had effectively gets its own submatrix (ie unique attention)

        B, S, D = x.shape

        if encoder_output is None:
            # running encoder
            Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # running decoder
            _, S_kv, _ = K.shape
            Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None and encoder_output is None:
            Q, K = self.rope(Q), self.rope(K)

        # Row: Query from [tok], Col: Key of [tok]
        # (nxn)

        raw_scores = Q @ K.transpose(-2, -1) # transpose (B, n, head_dim) => (B, head_dim, n)
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        if mask is not None:
            # decoder masked mha self-attention
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scaled_scores, dim=-1)

        out = attention @ V  # αv1 + αv2.. + αvn = contexualised word represntation based on attention
        out = out.transpose(1, 2) # from head -> [tokenlist] to token -> [headlist]
        out = out.contiguous() # memfix

        out = out.view(B, S, self.d_emb) # concat into nx512 matrix
        return self.W_o(out) # output is nx512


class GQA(nn.Module):
    def __init__(self, d_emb=512, num_heads=8, groups=2, rope=None):
        # groups = num of K/V 
        super().__init__()

        # groups @ 1=MQA, 8=MHA
        assert d_emb % num_heads == 0
        assert num_heads % groups == 0
        if groups == 1:
            print("Warning: GQA is equivalent to MQA with groups=1")
        elif groups == num_heads:
            print(f"Warning: GQA is equivalent to MHA with groups={num_heads}")

        self.d_emb = d_emb
        self.num_heads = num_heads
        self.head_dim = d_emb // num_heads
        self.groups = groups
        self.rope = rope # RoPE(head_dim) for C2, None otherwise
        
        self.W_q = nn.Linear(d_emb, d_emb)
        self.W_k = nn.Linear(d_emb, self.groups * self.head_dim)
        self.W_v = nn.Linear(d_emb, self.groups * self.head_dim)

        self.W_o = nn.Linear(d_emb, d_emb)

    def forward(self, x, mask=None, encoder_output=None):
        Q = self.W_q(x)

        if encoder_output is not None:
            # decoder cross attention
            K = self.W_k(encoder_output)
            V = self.W_v(encoder_output)
        else:
            # normal encoder attention
            K = self.W_k(x)
            V = self.W_v(x)

        B, S, D = x.shape

        if encoder_output is None:
            # running encoder
            # multiple head matrix transpose
            Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            # assign groups, transpose to same layout as Q
            K = K.view(B, S, self.groups, self.head_dim).transpose(1, 2)
            V = V.view(B, S, self.groups, self.head_dim).transpose(1, 2)
        else:
            # running decoder
            _, S_kv, _ = K.shape
            Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, S_kv, self.groups, self.head_dim).transpose(1, 2)
            V = V.view(B, S_kv, self.groups, self.head_dim).transpose(1, 2)

        if self.rope is not None and encoder_output is None:
            Q, K = self.rope(Q), self.rope(K)

        # duplicate K and V to match the number of heads
        K = K.repeat_interleave(self.num_heads // self.groups, dim=1)
        V = V.repeat_interleave(self.num_heads // self.groups, dim=1)
        # not repeat cuz that would be k0 k1 k0 k1 instead of k0 k0.. k1 k1

        raw_scores = Q @ K.transpose(-2, -1)
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        if mask is not None:
            # decoder masked mha self-attention
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scaled_scores, dim=-1)

        out = attention @ V
        out = out.transpose(1, 2)
        out = out.contiguous()

        out = out.view(B, S, self.d_emb)
        return self.W_o(out)
