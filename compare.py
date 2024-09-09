import torch

results = torch.load('kv_cache.dump')

for tlayer in range(len(results[0])):
    for k in range(2):
        print(results[0][tlayer][k].shape)
        if k % 2 == 0:
            print(f"comparing the {tlayer}'s K cache")
        else:
            print(f"comparing the {tlayer}'s V cache")


        d1 = results[0][tlayer][k].squeeze(0)
        d2 = results[1][tlayer][k].squeeze(0)
        for l in range(17):
            if torch.allclose(d1[l], d2[l], rtol=1e-5):
                print(f"{l}th token equal")
            else:
                print(f"{l}th token not equal")



