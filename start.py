# Copyright (c) 2025 Shrivatsa Murugan.
# Distributed under the GNU GPL V3.0 software license, see the accompanying
# file LICENSE or https://www.gnu.org/licenses/gpl-3.0.en.html

# --APLANA GPT--

from modules import Tokenizer, Embedder, CausalSelfAttention, Transformer, FeedForward, TextDataset
from AplanaGPT import AplanaGPT
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

os.system("cls" if os.name == "nt" else "clear")

print ("-- AplanaGPT Version 1.0 --")

model = AplanaGPT(
    vsize = 256,
    ctxlen = 256,
    dmodel = 256,
    nhead = 4,
    tnum = 4,
    dropout = 0.1
)

# model = torch.compile(model)

steps = 5000

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = CosineAnnealingLR(optim, T_max=steps, eta_min=1e-6)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = TextDataset(text, model.tok, model.ctxlen, 4)

for step in range(steps):
    xb, yb = dataset.getbatch()
    logits, loss = model.forward(xb, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print(model.generate("ROMEO:", 100))