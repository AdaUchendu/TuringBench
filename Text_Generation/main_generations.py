from Text_Generation.Grover import generate as gene
from Text_Generation.GPT2 import generate_articles
from Text_Generation.PPLM import generate
from Text_Generation.Transformers import generate_texts
from Text_Generation.FACEBOOK_FAIR import generate as gen
import argparse
import os
import sys
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='Generate Texts')
parser.add_argument('--generator', required=True, type=str, help='[ ]', default='ctrl')
opt = parser.parse_args()


data = pd.read_csv('../data/Titles.csv')
data = data[0:2]
def main_generations():

    headlines = data['Title']
    if opt.generator == 'ctrl' or opt.generator == 'openai-gpt' or opt.generator == 'xlnet' or \
            opt.generator == 'xlm' or opt.generator == 'transfo-xl':

        if opt.generator == 'ctrl':
            prompts = ["News " + headline for headline in headlines]
        else:
            prompts = [headline for headline in headlines]
        generated = generate_texts(model_type=opt.generator, model_name_or_path=opt.generator, prompts=prompts,
                                   length=450, stop_token=None, temperature=0.4, repetition_penalty=2.2,
                                   k=0, p=0.9, padding_text="", xlm_language="", seed=42,
                                   num_return_sequences=1, dump_json_file=f'./gen/{opt.generator}_generation.json')

    elif opt.generator == 'pplm':
        generated = generate(data['Title'])

    elif opt.generator == 'gpt2':
        generated = generate_articles(data['Title'])

    elif opt.generator == 'grover':
        generated = gene(data)

    elif opt.generator == "fair":
        generated = gen(data['Title'])

    else:
        print('Invalid Text Generator')
        generated = None

    return generated


print(f'Generating Articles with {opt.generator}')

generations = main_generations()
generations.to_csv(f'../Generations/{opt.generator}_articles.csv')


