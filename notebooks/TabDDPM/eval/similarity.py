import numpy as np
import pandas as pd
import os 
import kaleido
import plotly

import sys
sys.path.append('../..')
from utils import *

from sdmetrics.visualization import get_column_plot
import seaborn as sns

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport


def calculate_similarity():
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

    save_dir = f'eval/similarity/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    # Column-wise density estimation & Pair-wise column correlation
    print('CALCULATING COLUMN-WISE DENSITY ESTIMATION & PAIR-WISE COLUMN CORRELATION...')
    qual_report = QualityReport()
    qual_report.generate(new_real_data, new_syn_data, metadata)

    quality =  qual_report.get_properties()

    Shape = quality['Score'][0]
    Trend = quality['Score'][1]

    with open(f'{save_dir}/quality.txt', 'w') as f:
        f.write(f'{Shape}\n')
        f.write(f'{Trend}\n')

    Quality = (Shape + Trend) / 2

    shapes = qual_report.get_details(property_name='Column Shapes')
    trends = qual_report.get_details(property_name='Column Pair Trends')
    shapes.to_csv(f'{save_dir}/shape.csv')
    trends.to_csv(f'{save_dir}/trend.csv')

    print(f"Error rate (%) of column-wise density estimation {model.upper()}: {(1 - shapes['Score'].mean())*100:.3f} ± {shapes['Score'].std()*100:.3f}")
    print(f"Error rate (%) of pair-wise column correlation score {model.upper()}: {(1 - trends['Score'].mean())*100:.3f} ± {trends['Score'].std()*100:.3f}")

    fig = qual_report.get_visualization(property_name='Column Shapes')
    fig.show()
    fig.write_image(f'{save_dir}/column-wise-density-estimation.png')

    fig = qual_report.get_visualization(property_name='Column Pair Trends')
    fig.show()
    fig.write_image(f'{save_dir}/pair-wise-column-correlation.png')
    print('DONE!', '\n')

    # Column values distribution
    print('DRAW COLUMN VALUES DISTRIBUTIONS...')
    columns = new_real_data.columns
    new_columns = [info['idx_name_mapping'][
                      str(info['inverse_idx_mapping'][
                          str(i)])] 
                    for i in columns]
    new_real_data.columns = new_columns
    new_syn_data.columns = new_columns

    new_real_data['data_type'] = 'Real'
    new_syn_data['data_type'] = 'Synthetic'

    for num_idx_init in info['num_col_idx']:
      fig = get_column_plot(
          real_data=new_real_data,
          synthetic_data=new_syn_data,
          column_name=info['idx_name_mapping'][str(num_idx_init)],
          plot_type='distplot'
      )

      fig.show()
      fig.write_image(f"{save_dir}/distribution-of-{info['idx_name_mapping'][str(num_idx_init)]}.png")


    for cat_idx_init in info['cat_col_idx']:
      fig = get_column_plot(
          real_data=new_real_data,
          synthetic_data=new_syn_data,
          column_name=info['idx_name_mapping'][str(cat_idx_init)],
          plot_type='bar'
      )

      fig.show()
      fig.write_image(f"{save_dir}/distribution-of-{info['idx_name_mapping'][str(cat_idx_init)]}.png")
    print('DONE!')

    return {
        "Column Shapes Score, %": Shape*100,
        "Column Pair Trends Score, %": Trend*100, 
        "Overall Score (Average), %": (Shape+Trend)*100/2,
        "Error rate (%) of column-wise density estimation, %":(1 - shapes['Score'].mean())*100,
        "Error rate (%) of column-wise density estimation std, %":shapes['Score'].std()*100,
        "Error rate (%) of pair-wise column correlation score, %":(1 - trends['Score'].mean())*100,
        "Error rate (%) of pair-wise column correlation score std, %":trends['Score'].std()*100,
    }