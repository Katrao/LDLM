import torch
import torch.nn as nn
from modules import Tokenizer, Embedder, CausalSelfAttention, Transformer, FeedForward

class AplanaGPT(nn.Module):
    def __init__(self, vsize: int, ctxlen: int, dmodel: int, nhead: int, tnum: int, dropout: float):
        super().__init__()
        self.ctxlen = ctxlen
        self.tok = Tokenizer()
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
        self.eval()
        x = torch.tensor([self.tok.encode(prompt)])

        for _ in range(maxtokens):
            logits, _ = self.forward(x, x)
            nxtlogits = logits[:, -1, :]
            probs = torch.softmax(nxtlogits, dim=-1)
            nxtid = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, nxtid), dim=1)
        
        return self.tok.decode(x[0].tolist())

