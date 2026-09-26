# EncoderLayer
# DecoderLayer
# Encoder
# Decoder
# Transformer


import math

import torch
import torch.nn as nn

from .attention import *
from .ffn import *
from .norm import *
from .positional import *


class Transformer(nn.Module):
    def __init__(self, d_emb, src_itos, tgt_stoi, posconf, attconf, normconf, ffnconf, dropout=0.1, N=6):
        super().__init__()
        self.encoder = Encoder(d_emb, src_itos, posconf, attconf, normconf, ffnconf, dropout, N)
        self.decoder = Decoder(d_emb, tgt_stoi, posconf, attconf, normconf, ffnconf, dropout, N)
        
        self.proj = nn.Linear(d_emb, len(tgt_stoi))
        
    def forward(self, input_ids, output_ids):
        encoder_padmask = (input_ids != 0)[:, None, None, :]
        decoder_padmask = (output_ids != 0)[:, None, None, :]
        causalmask = torch.ones(
                output_ids.size(1), # output_ids = (B, S)
                output_ids.size(1), 
                dtype=torch.bool, 
                device=output_ids.device
            ).tril()[None, None] 
            # set everything above the L2R diagonal to False
            # corresponds row: current token col: tokens being attended to
            # so future tokens are set to False
            # [None, None] transforms to (1, 1, S, S) (for later use with (B, H, S, S) att. matrix format)
            
        encoder_output = self.encoder(input_ids, mask=encoder_padmask)
        decoder_output = self.decoder(output_ids, encoder_output, camask=encoder_padmask, samask=(decoder_padmask & causalmask))
        
        return self.proj(decoder_output)



class EncoderLayer(nn.Module):
    # the repeating block
    # takes input: output of positional embedding function (passed from Encoder)

    def __init__(self, attconf, normconf, ffnconf, dropout=0.1):
        # dropout=0.1 as per AIAYN
        super().__init__()

        self.att = attconf["class"](**attconf["args"])
        self.ffn = ffnconf["class"](**ffnconf["args"])

        # 1 = for attention bit
        # 2 = for FFN bit

        self.norm1 = normconf["class"](**normconf["args"])
        self.norm2 = normconf["class"](**normconf["args"])

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None): # usually padmask
        # pre norm instead of post norm (AIAYN uses post norm)

        # Prenorm, Attention, Dropout, Residual
        x = x + self.dropout1(self.att(self.norm1(x), mask=mask))

        # Prenorm, FFN, Dropout, Residual
        x = x + self.dropout2(self.ffn(self.norm2(x)))

        return x


class Encoder(nn.Module):
    def __init__(
        self, d_emb, itos, posconf, attconf, normconf, ffnconf, dropout=0.1, N=6
    ):
        super().__init__()

        # itos: map(index -> token)

        self.d_emb = d_emb
        self.embedding = nn.Embedding(
            len(itos), d_emb, padding_idx=0
        )  # token ID 0 is a padding token so this is okay
        self.itos = itos
        self.dropout = nn.Dropout(dropout)

        self.norm = normconf["class"](**normconf["args"])
        self.PositionalEncoder = posconf["class"](**posconf["args"])
        # pass the rest down to EncoderLayer cuz they need to be independent

        self.layers = nn.ModuleList(
            [EncoderLayer(attconf, normconf, ffnconf, dropout) for _ in range(N)]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_emb)  # Scale the embedding
        x = self.dropout(self.PositionalEncoder(x))
        for layer in self.layers:
            x = layer(x, mask=mask) # usually padmask for encoder
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, attconf, normconf, ffnconf, dropout=0.1):
        super().__init__()

        self.self_att = attconf["class"](**attconf["args"])
        self.cross_att = attconf["class"](**attconf["args"])
        self.ffn = ffnconf["class"](**ffnconf["args"])

        self.norm1 = normconf["class"](**normconf["args"])
        self.norm2 = normconf["class"](**normconf["args"])
        self.norm3 = normconf["class"](**normconf["args"])

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, camask=None, samask=None, encoder_output=None): 
        # cross-att mask, self-attn mask
        # usually for the decoder camask mask = padmask and samask = causal mask & padding mask

        # Prenorm, Masked Multi Head Attention, Dropout, Residual
        x = x + self.dropout1(self.self_att(self.norm1(x), mask=samask))

        # Prenorm, Cross Attention, Dropout, Residual
        x = x + self.dropout2(
            self.cross_att(self.norm2(x), encoder_output=encoder_output, mask=camask)
        )

        # Prenorm, FFN, Dropout, Residual
        x = x + self.dropout3(self.ffn(self.norm3(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, d_emb, stoi, posconf, attconf, normconf, ffnconf, dropout=0.1, N=6):
        super().__init__()

        self.d_emb = d_emb
        self.embedding = nn.Embedding(
            len(stoi), d_emb, padding_idx=0
        )
        self.stoi = stoi
        self.dropout = nn.Dropout(dropout)
        self.PositionalEncoder = posconf["class"](**posconf["args"])
        self.norm = normconf["class"](**normconf["args"])

        self.layers = nn.ModuleList(
            [DecoderLayer(attconf, normconf, ffnconf, dropout) for _ in range(N)]
        )


    def forward(self, x, encoder_output, camask=None, samask=None):
        # input x = token IDs with ID(<bos>) prepended
        
        x = self.embedding(x) * math.sqrt(self.d_emb)
        x = self.dropout(self.PositionalEncoder(x))

        for layer in self.layers:
            x = layer(x, camask=camask, samask=samask, encoder_output=encoder_output)

        return self.norm(x)