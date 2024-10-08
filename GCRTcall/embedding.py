
import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, dropout=None, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        self.is_encoder = True
        if dropout:
            self.dropout = torch.nn.Dropout(p=dropout)
            self.is_encoder = False
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length) -> Tensor:
        if self.is_encoder:
            return self.pe[:, :length]
        else:
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
            return self.dropout(x)
        
class Feature_extract(torch.nn.Module):
    def __init__(self, d_model, stride):
        super(Feature_extract, self).__init__()
        self.conv_ex = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 5, stride=1, padding=5//2, bias=True),
            torch.nn.BatchNorm1d(4),
            torch.nn.SiLU(),
            torch.nn.Conv1d(4, 16, 5, stride=1, padding=5//2, bias=True),
            torch.nn.BatchNorm1d(16),
            torch.nn.SiLU(),
            torch.nn.Conv1d(16, d_model, 19, stride=stride, padding=19//2, bias=True),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.SiLU(),
        )

    def forward(self, x):
        return self.conv_ex(x).permute(0, 2, 1)
    
class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab, pad):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(vocab+2, d_model, pad)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)