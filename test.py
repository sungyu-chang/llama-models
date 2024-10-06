import torch
import torch.nn.functional as F
import os
from fairscale.nn.model_parallel.layers import ColumnParallelLinear

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")

if not model_parallel_is_initialized():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

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

difference_mask = torch.abs(r3 - r4) > 1e-8

# Count the number of different elements
count = difference_mask.sum().item()

# Get the indices where the elements are different
different_indices = torch.nonzero(difference_mask, as_tuple=True)

# Print the different pairs of elements
print(f"There are {count} different elements.")
for index in zip(*different_indices):
    print(f"batch{index}: {r3[index]} vs sequence{index}: {r4[index]}")

print(f"count of difference is {count}")
if torch.allclose(r3, r4, rtol=1e-8):
    print("result is equal")
else:
    print("result is not equal")

linear_layer = torch.nn.Linear(4096, 4096, bias=False)

linear_layer.to(torch.bfloat16)
r1 = linear_layer(tensor_1)
r2 = linear_layer(tensor_2)

r4 = linear_layer(tensor_3)
r3 = torch.concat(( r1, r2 ), dim=1)

difference_mask = torch.abs(r3 - r4) > 1e-8

# Count the number of different elements
count = difference_mask.sum().item()

# Get the indices where the elements are different
different_indices = torch.nonzero(difference_mask, as_tuple=True)

# Print the different pairs of elements
print(f"There are {count} different elements.")
for index in zip(*different_indices):
    print(f"batch{index}: {r3[index]} vs sequence{index}: {r4[index]}")

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

difference_mask = torch.abs(r3 - r4) > 1e-8

# Count the number of different elements
count = difference_mask.sum().item()

# Get the indices where the elements are different
different_indices = torch.nonzero(difference_mask, as_tuple=True)

# Print the different pairs of elements
print(f"There are {count} different elements.")
for index in zip(*different_indices):
    print(f"batch{index}: {r3[index]} vs sequence{index}: {r4[index]}")
