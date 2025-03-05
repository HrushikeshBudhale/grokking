from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        # head projections
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_key, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)
        # scaled dot product
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.d_key ** 0.5
        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))
        attn = self.softmax(attn)
        return torch.matmul(attn, v)
    
    def __repr__(self) -> str:
        return f'AttentionHead(d_model={self.d_model}, d_key={self.d_key})'

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_key = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, d_key) for _ in range(n_heads)])
        self.linear = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        # parallel attention heads
        x = torch.cat([head(queries, keys, values, mask) for head in self.heads], dim=-1)
        return self.linear(x)

class FFN(nn.Module):
    def __init__(self, d_model: int, non_linearity:str='relu') -> None:
        super().__init__()
        d_ff = 4*d_model
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = {'relu': F.relu, 'gelu': F.gelu}[non_linearity]
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.act(self.linear1(x)))
    
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0, non_linearity: str='relu') -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, non_linearity=non_linearity)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        a1 = self.self_attn(x, x, x, mask)
        a1 = self.attn_norm(x+a1)
        a2 = self.dropout(self.ffn(a1))
        x = self.ffn_norm(a1+a2)
        return x
        
    def __repr__(self) -> str:
        return f'Block(d_model={self.d_model}, n_heads={self.n_heads})'
    
class Decoder(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, dropout: float=0.0, non_linearity: str='relu') -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout, non_linearity) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
        
    def __repr__(self) -> str:
        return f'Decoder(d_model={self.d_model}, n_blocks={self.n_blocks}, n_heads={self.n_heads})'


class Transformer(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 n_blocks: int = 4,
                 n_heads: int = 4,
                 dropout: float=0.1,
                 max_context_len: int = 1024,
                 vocab_size: int = 2000,
                 non_linearity: str='relu') -> None:
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.vocab_size = vocab_size
        self.non_linearity = non_linearity
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.register_buffer('positional_encoding', self._positional_encoding(max_context_len, d_model))
        self.register_buffer('mask', self.mask(max_context_len))
        
        self.decoder = Decoder(d_model, n_blocks, n_heads, dropout, non_linearity)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        
    @classmethod
    def _positional_encoding(cls, context_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(context_len, d_model)
        for pos in range(context_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))
        return pe.unsqueeze(0)

    @staticmethod
    def mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()
        
    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.positional_encoding[:context_len, :]
        return self.embedding(indices) + pe
    
    def forward(self, x: Tensor, pos: int = None) -> Tensor:
        x = x.to(self.embedding.weight.device)
        self_attn_mask = self.mask[:x.shape[-1], :x.shape[-1]]
        
        # Decode
        x = self.embed(x)
        decoded = self.decoder(decoded, self_attn_mask)
        
        # Return predictions for specific position
        if pos is not None:
            decoded = decoded[:, pos, :]
        y_hat = self.linear(decoded)
        return y_hat
    
    def __repr__(self) -> str:
        return f'Transformer(d_model={self.d_model}, n_blocks={self.n_blocks}, n_heads={self.n_heads})'