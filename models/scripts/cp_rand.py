import torch
from pathlib import Path
from termcolor import cprint
import torch.nn.functional as F

from .compare import compare_elt

rand_file = Path('rand.pt')
if rand_file.is_file():
    rand_gpu = torch.load('rand.pt', map_location='cuda')
    A = rand_gpu["A"]
    B = rand_gpu["B"]
    C = rand_gpu["C"]
else:
    A = torch.rand(4096, 4096, device='cuda', dtype=torch.bfloat16)
    B = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)
    C = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)

A1 = A.to('cpu')
B1 = B.to('cpu')
C1 = C.to('cpu')

def mulcat(W, B, C):
    return torch.concat(( F.linear(B, W), F.linear(C, W) ), dim=0)
def catmul(W, B, C):
    return F.linear(torch.concat((B, C), dim=0), W)
def mycp(W, B, C):
    compare_elt(mulcat(W, B, C), catmul(W, B, C))

cprint("compare two results from gpu")
mycp(A, B, C)
cprint("compare two results from cpu")
mycp(A1, B1, C1)

rand = {} 
rand["A"] = A
rand["B"] = B
rand["C"] = C
if not rand_file.is_file():
    torch.save(rand, "rand.pt")

data = torch.load("data.pt", map_location='cuda')
data_cpu = torch.load("data.pt", map_location='cpu')

mycp(data["w0"], data["x2"], data["x3"])
mycp(data["w1"], data["x2"], data["x3"])
mycp(data["w2"], data["x2"], data["x3"])
mycp(data_cpu["w0"], data_cpu["x2"], data_cpu["x3"])
mycp(data_cpu["w1"], data_cpu["x2"], data_cpu["x3"])
mycp(data_cpu["w2"], data_cpu["x2"], data_cpu["x3"])
