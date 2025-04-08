import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
import pickle

# Metrics
from sdmetrics import load_demo
from sdmetrics.single_table import LogisticDetection

from matplotlib import pyplot as plt

import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../..')
from utils import *

def calculate_detection():
    # CONFIG = Config('./config.json')
    dataname = CONFIG.get_arg('dataname')
    model = CONFIG.get_arg('method')

    if not CONFIG.get_arg('save_path'):
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = CONFIG.get_arg('save_path')
    if not CONFIG.get_arg('real_path'):
        real_path = f'synthetic/{dataname}/real.csv'
    else:
        real_path = CONFIG.get_arg('real_path')

    data_dir = f'data/{dataname}' 

    if not CONFIG.get_arg('info_path'):
        info_path = f'{data_dir}/info.json'
    else:
        info_path = CONFIG.get_arg('info_path')
    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    save_dir = f'eval/detection/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    # qual_report.generate(new_real_data, new_syn_data, metadata)

    score = LogisticDetection.compute(
        real_data=new_real_data,
        synthetic_data=new_syn_data,
        metadata=metadata
    )

    print(f'{dataname}, {model}: {score}')

    return {
        "Score": score
    }