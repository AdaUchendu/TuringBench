import pandas as pd
import numpy as np

from sources.Fairseq import *
import torch

en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
en_lm.cuda()

def generate(text):

  gen = []
  for i in text:
    gen.append(en_lm.sample(i, sampling=True, sampling_topk=100, temperature=0.8,
                            verbose=True, max_len_a=700, max_len_b=700, min_len=500))

  return pd.DataFrame({'Generation': gen})