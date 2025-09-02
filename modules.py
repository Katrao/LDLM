# Copyright (c) 2025 Shrivatsa Murugan.
# Distributed under the GNU GPL V3.0 software license, see the accompanying
# file LICENSE or https://www.gnu.org/licenses/gpl-3.0.en.html

import torch
import torch.nn as nn
from tokenizers import BertWordPieceTokenizer

class Tokenizer:
    def __init__(self, vsize: int = 1000):
        self.vocabsize = vsize
        self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        self.tokenizer.train(files=["tinyshakespeare.txt"], vocab_size=self.vocabsize, min_frequency=2)
    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids
    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)

class Embedder(nn.Module):
    def __init__(self, vsize: int, ctxlen : int, dmodel : int):
        super().__init__()
        self.tokemb = nn.Embedding(vsize, dmodel)
        self.posemb = nn.Embedding(ctxlen, dmodel)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        tvec = self.tokemb(x)
        pid = torch.arange(t, device=x.device)
        pvec = self.posemb(pid)[None, :, :]
        return self.dropout(tvec + pvec)
    
class CausalSelfAttention(nn.Module):
    def __init__(self, dmodel: int, nhead: int, ctxlen: int, dropout: float):
        super().__init__()
        assert dmodel % nhead == 0, "dmodel must be divisible by nhead"
        self.nhead = nhead
        self.hdim = dmodel // nhead

        self.qkv = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self.proj = nn.Linear(dmodel, dmodel)
        self.attentiondrop = nn.Dropout(dropout)
        self.residualdrop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(ctxlen, ctxlen)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.nhead, self.hdim).transpose(1, 2)
        k = k.view(b, t, self.nhead, self.hdim).transpose(1, 2)
        v = v.view(b, t, self.nhead, self.hdim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.hdim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attentiondrop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.residualdrop(self.proj(y))
    
class FeedForward(nn.Module):
    def __init__(self, dmodel: int, dropout : float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dmodel, 4 * dmodel),
            nn.GELU(),
            nn.Linear(4 * dmodel, dmodel),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, dmodel: int, nhead: int, ctxlen: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dmodel)
        self.attn = CausalSelfAttention(dmodel, nhead, ctxlen, dropout)
        self.ln2 = nn.LayerNorm(dmodel)
        self.ff = FeedForward(dmodel, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TextDataset:
    def __init__(self, text: str, tok: Tokenizer, ctxlen: int, bsize: int):
        self.tok = tok
        self.ctxlen = ctxlen
        self.bsize = bsize
        self.tokens = tok.encode(text)
    
    def getbatch(self) -> tuple:
        ix = torch.randint(0, len(self.tokens) - self.ctxlen, (self.bsize,))
        x = torch.stack([torch.tensor(self.tokens[i:i+self.ctxlen]) for i in ix])
        y = torch.stack([torch.tensor(self.tokens[i+1:i+self.ctxlen+1]) for i in ix])
        return x, y