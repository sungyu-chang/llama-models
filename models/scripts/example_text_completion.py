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
from termcolor import cprint


THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: int = 10,
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
    #     "The color of the sky is blue but sometimes it can also be",
    #     "The color of the sky is blue but sometimes it can also be"
    # ]
    prompts = [
        "The color of the sky is blue but sometimes it can also be",
        "The color of the sky is blue but sometimes it can also be green, red"
    ]
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
        torch.save(results, f'{k}-kv_cache.pt')
        print("\n==================================\n")



def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
