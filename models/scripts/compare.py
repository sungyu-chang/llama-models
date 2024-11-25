import torch
import torch.nn.functional as F
from termcolor import cprint

def compare_elt(t1, t2, print_index=True):

    difference_mask = torch.abs(t1 - t2) > 0
    count = difference_mask.sum().item()

# Get the indices where the elements are different
    different_indices = torch.nonzero(difference_mask, as_tuple=True)

    if print_index:
        for index in zip(*different_indices):
            print(f"t1{index}: {t1[index]} vs t2{index}: {t2[index]}")
    cprint(f"count of difference is {count}", "yellow")
    return count

# compare two tensor's column, which column is same which is not
# t1's shape is [bsz, token_len, 8, 128]
def compare_column(t1, t2, print_index: bool = True):
    t1 = t1.squeeze(0).reshape(-1, 1024)
    t2 = t2.squeeze(0).reshape(-1, 1024)
    equal = True
    for i in range(min(t1.shape[0],t2.shape[0])):
        result = []
        num_diff_elt = compare_elt(t1[i], t2[i], False)
        if num_diff_elt != 0:
            result.append(i)
        if print_index:
            print(result)
        if result:
            equal = False
    if equal:
        print("The two tensor is different")

def compare_dump_kvcache(t1, t2, print_index=True):
    if len(t1) != len(t2):
        cprint("tensor length different", color="red")
    for i in range(min(len(t1), len(t2))):
        print(f"comparing the {i}'s K cache")
        compare_column(t1[i][0], t2[i][0], print_index)
        print(f"comparing the {i}'s V cache")
        compare_column(t1[i][1], t2[i][1], print_index)

def compare(results, token_len: int, detail_print = False):


    if len(results[0]) != len(results[1]):
        cprint("the shape of two tensor is different", "red")
        return
    equal = True
    for tlayer in range(len(results[0])):
        for k in range(2):
            if detail_print:
                print(results[0][tlayer][k].shape)
            if detail_print:
                if k % 2 == 0:
                    print(f"comparing the {tlayer}'s K cache")
                else:
                    print(f"comparing the {tlayer}'s V cache")


            d1 = results[0][tlayer][k].squeeze(0)
            d2 = results[1][tlayer][k].squeeze(0)
            for tk_idx in range(token_len):
                if torch.allclose(d1[tk_idx], d2[tk_idx], rtol=0):
                    if detail_print:
                        print(f"{tk_idx + 1}th token equal")
                else:
                    if detail_print:
                        print(f"{tk_idx + 1}th token not equal")
                    equal = False

    if equal:
        cprint("KV is equal", "light_green")
    else:
        cprint("KV is not equal", "light_green")

# sequence processing, multiply then concat
def mulcat(W, x1, x2):
    return torch.concat(( F.linear(x1, W), F.linear(x2, W) ), dim=0)

# batch processing, concat then multiply
def catmul(W, x1, x2):
    return F.linear(torch.concat((x1, x2), dim=0), W)

# compare the batch and sequence results
def mycp(W, x1, x2, print_index=True):
    compare_elt(mulcat(W, x1, x2), catmul(W, x1, x2), print_index)

def compare_batch_sequence(w_cpu, x1_cpu, x2_cpu, w_gpu, x1_gpu, x2_gpu):
    cprint("==========================================", "blue")
    cprint("compare batch and sequence results from GPU", "green")
    mycp(w_gpu, x1_gpu, x2_gpu)
    cprint("==========================================", "blue")
    cprint("compare batch and sequence results from CPU", "green")
    mycp(w_cpu, x1_cpu, x2_cpu)

    cprint("==========================================", "blue")
    cprint("compare sequence results from GPU with CPU's results", "green")
    compare_elt(mulcat(w_gpu, x1_gpu, x2_gpu).cpu(), catmul(w_cpu, x1_cpu, x2_cpu))
    cprint("==========================================", "blue")
    cprint("compare batch results from GPU with CPU's results", "green")
    compare_elt(catmul(w_gpu, x1_gpu, x2_gpu).cpu(), catmul(w_cpu, x1_cpu, x2_cpu))
if __name__ == "__main__":
    results = torch.load('kv_cache.pt')
    from ..llama3.reference_impl.generation import dump_layer
    compare(results, dump_layer, True)

