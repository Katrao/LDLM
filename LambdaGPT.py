# Copyright (c) 2025 Shrivatsa Murugan.
# Distributed under the GNU GPL V3.0 software license, see the accompanying
# file LICENSE or https://www.gnu.org/licenses/gpl-3.0.en.html

import torch
import torch.nn as nn
from modules import Tokenizer, Embedder, CausalSelfAttention, Transformer, FeedForward

class LambdaGPT(nn.Module):
    def __init__(self, vsize: int, ctxlen: int, dmodel: int, nhead: int, tnum: int, dropout: float):
        super().__init__()
        self.ctxlen = ctxlen
        self.tok = Tokenizer(vsize)
        self.emb = Embedder(vsize, ctxlen, dmodel)
        self.tblocks = nn.ModuleList([Transformer(dmodel, nhead, ctxlen, dropout) for _ in range(tnum)])
        self.lnf = nn.LayerNorm(dmodel)
        self.head = nn.Linear(dmodel, vsize, bias=False)
    
    def forward(self, x: torch.Tensor, exp) -> torch.Tensor:
        x = self.emb(x)
        for block in self.tblocks:
            x = block(x)
        x = self.lnf(x)
        logits = self.head(x)

        loss = None
        if exp is not None:
            b, t, v = logits.shape
            loss = nn.functional.cross_entropy(logits.view(b * t, v), exp.view(b * t))
        
        return logits, loss
    
    def generate(self, prompt: str, maxtokens: int) -> str:
        with torch.no_grad():    
            self.eval()
            x = torch.tensor([self.tok.encode(prompt)])
            y = x.clone()
            for _ in range(maxtokens):
                logits, _ = self.forward(x, x)
                nxtlogits = logits[:, -1, :]
                probs = torch.softmax(nxtlogits, dim=-1)
                nxtid = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, nxtid), dim=1)
                y = torch.cat((y, nxtid), dim=1)
                if x.shape[1] > self.ctxlen:
                    x = x[:, 1:]
                print(self.tok.decode([nxtid.item()]), end="", flush=True)
        print()
        return self.tok.decode(y[0].tolist())

