import numpy as np
import pandas as pd
import os 
from sdv.metrics.tabular import BinaryDecisionTreeClassifier, BinaryLogisticRegression
import json
import sys
sys.path.append('../..')
from utils import *

def compute_loss(results, dataset, metric):
  control = results[dataset][metric]
  synth = results[dataset]['Synth_' + metric]

  return 100 - (synth/control * 100)


def calculate_base_metrics(make_binary, value):
    # CONFIG = Config('./config.json')
    dataname = CONFIG.get_arg('dataname')
    model = CONFIG.get_arg('method')

    if not CONFIG.get_arg('sample_save_path'):
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = CONFIG.get_arg('sample_save_path')

    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'

    data_dir = f'data/{dataname}' 

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    save_dir = f'eval/base_metrics/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # real_data.columns = range(len(real_data.columns))
    # test_data.columns = range(len(test_data.columns))
    # syn_data.columns = range(len(syn_data.columns))

    if make_binary:
      real_data[real_data.columns[-1]] = (real_data[real_data.columns[-1]]==value).astype(int)
      test_data[test_data.columns[-1]] = (test_data[test_data.columns[-1]]==value).astype(int)
      syn_data[syn_data.columns[-1]] = (syn_data[syn_data.columns[-1]]==value).astype(int)

    target_col = real_data.columns[-1]
    tree = BinaryDecisionTreeClassifier.compute(test_data, real_data, target=target_col)
    lr = BinaryLogisticRegression.compute(test_data, real_data, target=target_col)

    results = {}
    results[dataname] = {}
    results[dataname]['Tree'] = tree
    results[dataname]['LR'] = lr

    tree = BinaryDecisionTreeClassifier.compute(test_data, syn_data, target=target_col)
    lr = BinaryLogisticRegression.compute(test_data, syn_data, target=target_col)

    results[dataname]['Synth_Tree'] = tree
    results[dataname]['Synth_LR'] = lr

    print(f'{model.upper()}: {dataname.upper()} Dataset')
    print(f"Original Logistic: {results[dataname]['LR']:.3f}")
    print(f"Synthetic Logistic: {results[dataname]['Synth_LR']:.3f}")

    print(f"Original Tree: {results[dataname]['Tree']:.3f}")
    print(f"Synthetic Tree: {results[dataname]['Synth_Tree']:.3f}", '\n')

    results[dataname]['Accuracy Loss Tree'] = compute_loss(results, dataname, 'Tree')
    results[dataname]['Accuracy Loss LR'] = compute_loss(results, dataname, 'LR')

    print(f'{model.upper()} Accuracy Loss: {dataname.upper()}')
    print(f"Logistic: {results[dataname]['Accuracy Loss LR']:.3f}%")
    print(f"Tree: {results[dataname]['Accuracy Loss Tree']:.3f}%", '\n')
    
    pd.DataFrame(results).to_csv(f'{save_dir}/results.csv')