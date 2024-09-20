import torch


def compare(token_len: int, detail_print = False):
    results = torch.load('kv_cache.pt')

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
                if torch.allclose(d1[tk_idx], d2[tk_idx], rtol=1e-8):
                    if detail_print:
                        print(f"{tk_idx}th token equal")
                else:
                    if detail_print:
                        print(f"{tk_idx}th token not equal")
                    equal = False

    if equal:
        print("KV is equal")
    else:
        print("KV is not equal")

if __name__ == "__main__":
    compare(15, True)

