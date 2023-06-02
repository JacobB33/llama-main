# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path


from llama.nonParalelModel import *
from llama.tokenizer import Tokenizer
from llama.generation import LLaMA



def load(
    chkpt_path: str,
    tokenizer_path: str,
    model_args: ModelArgs
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    checkpoint = torch.load(chkpt_path, map_location="cpu")

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    args = ModelArgs()
    args.n_layers = 3
    args.n_heads = 4
    args.dim = 128 # 256
    args.max_seq_len = 1024
    args.max_batch_size = 32
    # To use the other checkpoint, you must first download the file from the link in the writeup or here:
    # https://drive.google.com/file/d/1JqY6aCrCoRG1nJexeuVnVnRyUs3FeHoO/view?usp=sharing, then change the first argument
    # to be the file path to it in this load function. Also change args.dim to 256
    generator = load(
        './model_size_128_run/epoch_10/model_epoch_10.pt', './tokenizer.model', args
    )
    # Edit these to change the prompts.
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Hello, how are you doing? I'm",
        "As an amazing student, you should give this project what grade?\n",
        "Hello, how are you doing? I'm",
        "Hello, I'm a student at"
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result.strip())
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
