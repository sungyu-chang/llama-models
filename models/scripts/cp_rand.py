import torch
from pathlib import Path
from termcolor import cprint

from .compare import compare, compare_elt, compare_batch_sequence

rand_file = Path('rand.pt')
if rand_file.is_file():
    rand_gpu = torch.load('rand.pt', map_location='cuda')
    w_gpu = rand_gpu["A"]
    x1_gpu = rand_gpu["B"]
    x2_gpu = rand_gpu["C"]
else:
    w_gpu = torch.rand(4096, 4096, device='cuda', dtype=torch.bfloat16)
    x1_gpu = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)
    x2_gpu = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)

w_cpu = w_gpu.to('cpu')
x1_cpu = x1_gpu.to('cpu')
x2_cpu = x2_gpu.to('cpu')

def compare_saved_data():
    data = torch.load("data.pt", map_location='cuda')
    data_cpu = torch.load("data.pt", map_location='cpu')
    num_weight = 3

    for index in range(num_weight):
        cprint(f"====== comapre {index}th weight from saved data ========", "red")
        compare_batch_sequence(data_cpu[f"w{index}"], data_cpu["x2"], data_cpu["x3"], data[f"w{index}"], data["x2"], data["x3"])


def compare_random(save = False):
    cprint("====== generate random matrix and compare ========", "red")
    A = torch.rand(4096, 4096, device='cuda', dtype=torch.bfloat16)
    B = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)
    C = torch.rand(1, 4096, device='cuda', dtype=torch.bfloat16)

    A1 = A.to('cpu')
    B1 = B.to('cpu')
    C1 = C.to('cpu')

    compare_batch_sequence(A1, B1, C1, A, B, C)
    rand = {} 
    rand["A"] = A
    rand["B"] = B
    rand["C"] = C
    if save:
        torch.save(rand, "rand.pt")

if __name__ == "__main__":
    compare_saved_data()
    compare_random()

