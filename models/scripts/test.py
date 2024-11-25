import torch
import torch.nn.functional as F
import os
from fairscale.nn.model_parallel.layers import ColumnParallelLinear
from termcolor import cprint
from .compare import compare, compare_elt
import random
import numpy as np
seed = 42

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("gloo")

if not model_parallel_is_initialized():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device('cpu')

cprint("test softmax", "light_yellow")
tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
tensor_1d_inf = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, -float('inf')])
print(F.softmax(tensor_1d, dim=0))
print(F.softmax(tensor_1d_inf, dim=0))


tensor_1 = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
tensor_2 = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
tensor_3 = torch.concat(( tensor_1, tensor_2 ), dim=1)


random_matrix = torch.randn(4096, 4096, dtype=torch.bfloat16)

r1 = torch.matmul(tensor_1, random_matrix)
r2 = torch.matmul(tensor_2, random_matrix)

r3 = torch.concat(( r1, r2 ), dim=1)

r4 = torch.matmul(tensor_3, random_matrix)
cprint("compare matmul", "light_yellow")
compare_elt(r3, r4)

linear_layer = torch.nn.Linear(4096, 4096, bias=False)

linear_layer.to(torch.bfloat16)
r1 = linear_layer(tensor_1)
r2 = linear_layer(tensor_2)

r4 = linear_layer(tensor_3)
r3 = torch.concat(( r1, r2 ), dim=1)

cprint("compare linear layer", "light_yellow")
compare_elt(r3, r4)

wk = ColumnParallelLinear(
    4096,
    8 * 128,
    bias=False,
    gather_output=False,
    init_method=lambda x: x,
)
wk.to(torch.bfloat16)

r1 = wk(tensor_1)
r2 = wk(tensor_2)
r4 = wk(tensor_3)

r3 = torch.concat(( r1, r2 ), dim=1)

cprint("compare ColumnParallelLinear", "light_yellow")
compare_elt(r3, r4)

result = torch.load("rt_dump.pt")

cprint("test w dump", "red")

x1 = result["x"][0]

x2 = result["x"][1]
x3 = result["x"][2]

# x1_reshape = x1.view(2, 4096)
# x2_reshape = x2.view(1, 4096)
# x3_reshape = x3.view(1, 4096)
# w0 = result["w"][0]
# w1 = result["w"][1]
# w2 = result["w"][2]
#
# data = {
#     "x1": x1_reshape,
#     "x2": x2_reshape,
#     "x3": x3_reshape,
#     "w0": w0,
#     "w1": w1,
#     "w2": w2,
# }
# torch.save(data, "data.pt")
cprint("compare the input", "yellow")
compare_elt(x1, torch.concat((x2, x3), dim = 1))

breakpoint()
cpuresult = {}
for i in range(3):
    w = result["w"][i]

    batch = F.linear(x1, w, None)

    sequence = torch.concat((F.linear(x2, w, None), F.linear(x3, w, None)), dim = 1)
    cprint("compare cpu result", "red")
    cprint(batch, "green")
    cprint(sequence, "blue")
    compare_elt(batch, sequence)
    if i == 0:
        cpuresult["xq"] = [batch, sequence]
    if i == 1:
        cpuresult["xk"] = [batch, sequence]
    if i == 2:
        cpuresult["xv"] = [batch, sequence]


x1 = x1.cuda()
x2 = x2.cuda()
x3 = x3.cuda()
gpuresult = {}
for i in range(3):
    w = result["w"][i].to("cuda")

    batch = F.linear(x1, w, None)

    sequence = torch.concat((F.linear(x2, w, None), F.linear(x3, w, None)), dim = 1)
    cprint("compare gpu result", "red")
    cprint(batch, "green")
    cprint(sequence, "blue")
    compare_elt(batch, sequence)
    if i == 0:
        gpuresult["xq"] = [batch.cpu(), sequence.cpu()]
    if i == 1:
        gpuresult["xk"] = [batch.cpu(), sequence.cpu()]
    if i == 2:
        gpuresult["xv"] = [batch.cpu(), sequence.cpu()]

cprint("compare batch result of xq", "red")
compare_elt(cpuresult["xq"][0], result["xq"][0])

cprint("compare sequence result of xq", "red")
compare_elt(cpuresult["xq"][1], torch.concat(( result["xq"][1], result["xq"][2] ), dim = 1))

cprint("compare batch result of xk", "red")
compare_elt(cpuresult["xk"][0], result["xk"][0])

cprint("compare sequence result of xk", "red")
compare_elt(cpuresult["xk"][1], torch.concat(( result["xk"][1], result["xk"][2] ), dim = 1))

cprint("compare batch result of xv", "red")
compare_elt(cpuresult["xv"][0], result["xv"][0])

cprint("compare sequence result of xv", "red")
compare_elt(cpuresult["xv"][1], torch.concat(( result["xv"][1], result["xv"][2] ), dim = 1))


for i in cpuresult:
    cprint(f"compare {i} batch result", "red")
    compare_elt(cpuresult[i][0], gpuresult[i][0])
    cprint(f"compare {i} sequence result", "red")
    compare_elt(cpuresult[i][1], gpuresult[i][1])

torch.save(cpuresult, 'cpu_result_es1-1.pt')
torch.save(gpuresult, 'gpu_result_es1-1.pt')
