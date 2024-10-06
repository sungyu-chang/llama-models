# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from pathlib import Path
from typing import Optional

import fire
import torch

from models.llama3.reference_impl.generation import Llama
from models.llama3.reference_impl.model import current_result
from termcolor import cprint
from compare import compare

THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: int = 4,
    model_parallel_size: Optional[int] = None,
):
    tokenizer_path = str(THIS_DIR.parent / "llama3/api/tokenizer.model")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    old_prompts = [
        "The color of the sky is blue but sometimes it can also be",
        """\
apple is pomme,
bannana is banane,
cherry is""",
        "1, 2, 3, 5, 8, 13",
        "ba ba black sheep, have you any wool?",
    ]
    # prompts = [
    #     "The color of the sky is blue but sometimes it can also be green, red",
    #     "The color of the sky is blue but sometimes it can also be green, red",
    # ]
    prompts = [
        "The color of the sky is blue but sometimes it can also be green, red",
        "The color of the sky is blue but sometimes it can also",
    ]
    # prompts = [
    #     "The color of the sky is blue but sometimes it can also be a little bit",
    #     "The color of the sky is blue but sometimes it can also be",
    # ]
    torch.set_printoptions(edgeitems=10)
    results = []
    for k, prompt in enumerate(prompts):
        tensor_result = []
        result = generator.text_completion(
            prompt,
            hex_result=tensor_result,
            temperature=0,
            top_p=0.9,
            max_gen_len=max_gen_len,
            logprobs=False,
        )
        results.append(tensor_result)

        cprint(f"{prompt}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")

    for key, value in current_result.items():
        print(key)
        batch = value[0]
        print(value[0].shape)
        print(value[1].shape)
        print(value[2].shape)
        sequence = torch.concat(( value[1], value[2] ), dim=1)
        
        print(batch[:, 0, :])
        print(value[1])
        print(batch[:, 1, :])
        print(value[2])
        cprint("first row", "green")
        difference_mask = torch.abs(value[0][:, 0, :] - value[1]) > 1e-8

        # Count the number of different elements
        count = difference_mask.sum().item()

        # Get the indices where the elements are different
        different_indices = torch.nonzero(difference_mask, as_tuple=True)

        for index in zip(*different_indices):
            print(f"batch{index}: {batch[index]} vs sequence{index}: {sequence[index]}")
        print(f"count of difference is {count}")

        cprint("second row", "green")

        difference_mask = torch.abs(value[0][:, 1, :] - value[2]) > 1e-8

        # Count the number of different elements
        count = difference_mask.sum().item()

        # Get the indices where the elements are different
        different_indices = torch.nonzero(difference_mask, as_tuple=True)

        print(f"There are {count} different elements.")
        # for index in zip(*different_indices):
        #     print(f"batch{index}: {batch[index]} vs sequence{index}: {sequence[index]}")

        # if torch.allclose(batch[:, 1, :], value[2], rtol=0):
        #     print(f"{key} is equal")
        # else:
        #     print(f"{key} is not equal")


    # dump the KV cache
    torch.save(results, 'kv_cache.pt')
    compare(15)


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
