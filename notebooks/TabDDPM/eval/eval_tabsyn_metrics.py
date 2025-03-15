import numpy as np
import pandas as pd
import os 
import kaleido
import plotly

from sdmetrics.visualization import get_column_plot
import seaborn as sns

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 
    print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    save_dir = f'eval/tabsyn_metrics/{dataname}/{model}'
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

    fig = qual_report.get_visualization(property_name='Column Shapes')
    fig.show()
    fig.write_image(f'{save_dir}/column-wise-density-estimation.png')

    fig = qual_report.get_visualization(property_name='Column Pair Trends')
    fig.show()
    fig.write_image(f'{save_dir}/pair-wise-column-correlation.png')
    print('DONE!')

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
      g = sns.catplot(data=pd.concat([new_real_data, new_syn_data]), 
                  x=info['idx_name_mapping'][str(cat_idx_init)], kind="count", hue='data_type')
      g.set_xticklabels(rotation=90) 
      g.figure.savefig(f"{save_dir}/distribution-of-{info['idx_name_mapping'][str(cat_idx_init)]}.png")
    print('DONE')