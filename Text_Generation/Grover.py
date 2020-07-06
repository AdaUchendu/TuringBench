import pandas as pd
import numpy as np
from sources.Grover import *
import os
import subprocess

def convert_data_format(data):
    data.rename(columns = {'Links': 'url', 'Title': 'title', 'Summary': 'summary', 'Article': 'text',
                           'Pub_Date': 'publish_date', 'Domain': 'domain', 'Authors': 'authors'})
    data['url_used'] = data['url']

    return data




def generate(data):
    conv_data = convert_data_format(data)
    conv_data.to_json('../data/conv_data.json', orient='records', lines=True)

    # return subprocess.call(['./run_grover.sh'])

    return os.system('PYTHONPATH=$(pwd) python sources/Grover/sample/contextual_generate.py '
                      '-model_config_fn sources/Grover/lm/configs/mega.json '
                      '-model_ckpt models/mega/model.ckpt -metadata_fn ../data/conv_data.json '
                      '-out_fn ../generations/grover.json')





# SHELL code.
# PYTHONPATH=$(pwd) python sources/Grover/sample/contextual_generate.py -model_config_fn
# sources/Grover/lm/configs/mega.json -model_ckpt
# models/mega/model.ckpt -metadata_fn ../data/conv_data.json -out_fn ../generations/grover.json