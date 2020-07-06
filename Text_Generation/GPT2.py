from sources.GPT2_Wrapper import *
import pandas as pd

import os
import sys
import torch
import random
import argparse
import numpy as np
from sources.GPT2_Wrapper.GPT2.model import (GPT2LMHeadModel)
from sources.GPT2_Wrapper.GPT2.utils import load_weight
from sources.GPT2_Wrapper.GPT2.config import GPT2Config
from sources.GPT2_Wrapper.GPT2.sample import sample_sequence
from sources.GPT2_Wrapper.GPT2.encoder import get_encoder


def text_generator(state_dict, input_text):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--quiet", type=bool, default=False)
    # parser.add_argument("--nsamples", type=int, default=1)
    # parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    # parser.add_argument("--batch_size", type=int, default=-1)
    # parser.add_argument("--length", type=int, default=-1)
    # parser.add_argument("--temperature", type=float, default=0.7)
    # parser.add_argument("--top_k", type=int, default=40)
    # args = parser.parse_args()

    # if args.quiet is False:
    #     print(args)

    # if args.batch_size == -1:
    #     args.batch_size = 1
    # assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    args_length = config.n_ctx // 2
    # if args.length == -1:
    #     args.length = config.n_ctx // 2
    # elif args.length > config.n_ctx:
    #     raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(input_text)
    context_tokens = enc.encode(input_text)

    generated = 0
    output_text = []
    for _ in range(1):
        out = sample_sequence(
            model=model, length=args_length,
            context=context_tokens if not True else None,
            start_token=enc.encoder['<|endoftext|>'] if True else None,
            batch_size=1,
            temperature=0.7, top_k=50, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(1):
            generated += 1
            text = enc.decode(out[i])
            # if args.quiet is False:
            #     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
            output_text.append(text)
    return output_text



def generate_articles(Titles):
    if os.path.exists('../sources/GPT2_Wrapper/gpt2-pytorch_model.bin'):
        state_dict = torch.load('../sources/GPT2_Wrapper/gpt2-pytorch_model.bin',
                                map_location='cpu' if not torch.cuda.is_available() else None)

        output_text = []
        for i in range(len(Titles)):
            output_text.append(text_generator(state_dict, Titles[i]))

        return pd.DataFrame({'Generations': output_text})