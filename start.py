# Copyright (c) 2025 Shrivatsa Murugan.
# Distributed under the GNU GPL V3.0 software license, see the accompanying
# file LICENSE or https://www.gnu.org/licenses/gpl-3.0.en.html

# --LAMBDA GPT--

from modules import TextDataset
from LambdaGPT import LambdaGPT
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import profile, record_function, ProfilerActivity
import os
import math

os.system("cls" if os.name == "nt" else "clear")

print ("-- LambdaGPT Version 1.0 --")

model = LambdaGPT(
    vsize = 1000,
    ctxlen = 256,
    dmodel = 256,
    nhead = 4,
    tnum = 4,
    dropout = 0.01
)

# model = torch.compile(model)

steps = 50000
wsteps = 1000

def lrlambda(step):
    if step < wsteps:
        return (step + 1) / wsteps
    progress = (step - wsteps) / (steps - wsteps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = LambdaLR(optim, lr_lambda=lrlambda)

dset = "tinyshakespeare.txt"
dsetn = "tinyshakespeare"

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = TextDataset(
    text = text, 
    tok = model.tok, 
    ctxlen = model.ctxlen, 
    bsize = 16
)

cdir = "checkpoints"

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_training"):
        for step in range(steps + 1):
            torch.set_num_threads(torch.get_num_threads())
            xb, yb = dataset.getbatch()
            logits, loss = model.forward(xb, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}                ")
            else:
                print(f"    Step {step}, Loss: {loss.item():.4f}", end="\r")
            if step % 1000 == 0 and step != 0:
                torch.save({
                    "step": step,
                    "mstate": model.state_dict(),
                    "ostate": optim.state_dict(),
                    "sstate": sched.state_dict() if sched is not None else None,
                    "loss": loss.item()
                }, os.path.join(cdir, f"lambdagpt_{dsetn}_{step}.pt"))
                print(f"Checkpoint saved at step {step}.")

    


print()
print()
model.generate("ROMEO:", 500)
print()
print()
model.generate("JULIET:", 500)
print()
print()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))