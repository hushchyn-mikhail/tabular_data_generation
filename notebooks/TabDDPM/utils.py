import os
import numpy as np
import pandas as pd
from urllib import request
import shutil
import zipfile
import sys
import json
import argparse
from torch.utils.data import Dataset
import sklearn.preprocessing

import src
from pprint import pprint

DATA_DIR = 'data'
NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip'
}
TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

# CONFIG
class Config():
  def __init__(self, path):
    with open(path, 'r') as file:
        self.config_global = json.load(file)
    
    for var in ['path', 'model_name']:
        assert self.config_global.get('path'), 'No path in config'
    
    self.path = self.config_global['path']
    self.model_name = self.config_global['model_name']
    
    if not os.path.isfile(self.path):
        self.config = {}
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
    else:
        with open(self.path, 'r') as file:
            self.config = json.load(file)
            
  def add_arg(self, param_name, param_value):
    self.config[param_name] = param_value
    with open(self.path, 'w', encoding='utf-8') as f:
        json.dump(self.config, f, ensure_ascii=False, indent=4)

  def get_arg(self, param_name):
    return self.config.get(param_name)
  
  def get_all_args(self):
    return self.config

  def delete_arg(self, param_name):
    self.config.pop(param_name)
    with open(self.path, 'w', encoding='utf-8') as f:
        json.dump(self.config, f, ensure_ascii=False, indent=4)

CONFIG = Config('./config.json')

# DOWNLOAD DATASET
def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name, name_to_save=None):
    print(f'Start processing dataset {name} from UCI.')
    if not name_to_save:
        name_to_save = name
    save_dir = f'{DATA_DIR}/{name_to_save}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

def download_dataset(name, name_to_save=None):
    download_from_uci(name, name_to_save)
    
# PROCESS DATASET
INFO_PATH = 'data/Info'

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]


        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]



        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    


def process_data(name):

    if name == 'news':
        preprocess_news()
    elif name == 'beijing':
        preprocess_beijing()

    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['data_path']
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:     
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path, header = None)
        train_df = data_df

    else:  
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data_df.shape)

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()


    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

 
    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    if X_cat_train.shape[1] > 0:
        np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
        print("X_cat_train have not been saved. No data.")
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    if X_cat_test.shape[1] > 0:
        np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
        print("X_cat_test have not been saved. No data.")
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)


    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
    
    train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)

def categorial_to_OHE(name, do_quantile_and_standart_scale=False):
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)
    
    save_dir = f'data/{name}'
    
    if info['test_path']:
        # if testing data is given
        test_path = info['test_path']
    
        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:     
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')
    
    info['column_names_initial'] = [*np.array(info['column_names'])[info['num_col_idx']], 
                                *np.array(info['column_names'])[info['cat_col_idx']],
                                *np.array(info['column_names'])[info['target_col_idx']]].copy()
    
    info['num_col_idx_initial'] = list(range(len(info['num_col_idx'])))
    info['cat_col_idx_initial'] = list(range(len(info['num_col_idx']), 
                                                 len(info['num_col_idx']) + len(info['cat_col_idx'])))
    info['target_col_idx_initial'] = list(range(len(info['num_col_idx']) + len(info['cat_col_idx']), 
                                                    len(info['num_col_idx']) + len(info['cat_col_idx']) + len(info['target_col_idx'])))
    
    train_tmp = pd.read_csv(info['data_path'], header = None)
    test_tmp = pd.read_csv(test_save_path, header = None)
    tmp = pd.concat([train_tmp, test_tmp])
    
    real_num_data = tmp[info['num_col_idx']]
    real_cat_data = tmp[info['cat_col_idx']]
    real_target_data = tmp[info['target_col_idx']]
    
    concat_tmp = pd.concat([real_num_data,
                 real_cat_data, 
                 real_target_data], axis=1)
    concat_tmp.columns = list(range(concat_tmp.shape[1]))
    
    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
        
    concat_tmp.iloc[:len(train_tmp)].to_csv(f'synthetic/{name}/initial_real.csv', 
                                                                         index = False, 
                                                                         header=info['column_names_initial'])
    concat_tmp.iloc[-len(test_tmp):].to_csv(f'synthetic/{name}/initial_test.csv', 
                                                                         index = False, 
                                                                         header=info['column_names_initial'])
    
    ohe_cat_data = pd.get_dummies(real_cat_data).astype(int)
    ohe_target_data = pd.get_dummies(real_target_data).astype(int)
    len_num_prev = real_num_data.shape[1]
    len_cat_prev = ohe_cat_data.shape[1]
    len_target_prev = ohe_target_data.shape[1] - 1
    
    tmp = pd.concat([real_num_data, 
                 ohe_cat_data, 
                 ohe_target_data], axis=1)
    info['initial_column_names'] = list(tmp.columns)
    tmp = tmp.drop(columns=tmp.columns[-1])

    info['column_names'] = list([str(col) for col in tmp.columns])
    info['column_info'] = {}
    for col in info['column_names']:
        info['column_info'][col] = 'float'
    
    tmp.columns = list(range(tmp.shape[1]))
    
    info['target_col_idx'] = list(tmp.columns[-1:])
    info['num_col_idx'] = list(tmp.columns[:-1])
    info['cat_col_idx'] = []
    info['prev_cat_num'] = len_cat_prev + len_target_prev

    if do_quantile_and_standart_scale:
        train_num_if = tmp[range(len_num_prev)].iloc[:len(train_tmp)]
        test_num_if = tmp[range(len_num_prev)].iloc[-len(test_tmp):]
        
        train_cat_if = tmp[range(len_num_prev, tmp.shape[1])].iloc[:len(train_tmp)]
        test_cat_if = tmp[range(len_num_prev, tmp.shape[1])].iloc[-len(test_tmp):]
    
        # QUANTILE FOR NUMERICAL
        num_normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(train_num_if.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=0,
            )
        num_normalizer.fit(train_num_if)
    
        train_num_if = pd.DataFrame(num_normalizer.transform(train_num_if), columns=train_num_if.columns)
        test_num_if = pd.DataFrame(num_normalizer.transform(test_num_if), columns=test_num_if.columns)
    
        # SCANDART SCALER FOR OHE
        cat_normalizer = sklearn.preprocessing.StandardScaler()
        cat_normalizer.fit(train_cat_if)
    
        train_cat_if = pd.DataFrame(cat_normalizer.transform(train_cat_if), columns=train_cat_if.columns)
        test_cat_if = pd.DataFrame(cat_normalizer.transform(test_cat_if), columns=test_cat_if.columns)
    
        tmp = pd.concat([pd.concat([train_num_if, train_cat_if], axis=1), pd.concat([test_num_if, test_cat_if], axis=1)])
        tmp = tmp.reset_index(drop=True)
    
    tmp.iloc[:len(train_tmp)].to_csv(info['data_path'], index=False, header=None)
    tmp.iloc[-len(test_tmp):].to_csv(test_save_path, index=False, header=None)
    info['done_ohe_noise'] = 'yes'
    
    info['task_type'] = 'regression'


    pprint(info)
    
    with open(f'{INFO_PATH}/{name}.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    
    print("DONE CONVERTING DATA!")
    if do_quantile_and_standart_scale:
        return {
            'num_normalizer':num_normalizer,
            'cat_normalizer':cat_normalizer,
            'len_num_prev':len_num_prev,
            'len_cat_prev':len_cat_prev,
            'len_target_prev':len_target_prev
        }

def postprocess_OHE(name, name_copy):
    with open(f'./data/{name}/info.json', 'r') as f:
        info = json.load(f)
    
    with open(f'./data/{name_copy}/info.json', 'r') as f:
        info_copy = json.load(f)
    
    initial_info = info.copy()
    
    initial_info['task_type'] = 'binclass'
    initial_info['column_names'] = info['column_names_initial']
    initial_info['num_col_idx'] = info['num_col_idx_initial']
    initial_info['cat_col_idx'] = info['cat_col_idx_initial']
    initial_info['target_col_idx'] = info['target_col_idx_initial']
    initial_info['idx_name_mapping'] = dict(zip(range(len(info['column_names_initial'])), info['column_names_initial']))
    
    initial_info['metadata']['columns'] = {}
    for i in initial_info['num_col_idx']:
        initial_info['metadata']['columns'][str(i)] = {'sdtype': 'numerical', 'computer_representation': 'Float'}
    for i in initial_info['cat_col_idx']:
        initial_info['metadata']['columns'][str(i)] = {'sdtype': 'categorical'}
    for i in initial_info['target_col_idx']:
        initial_info['metadata']['columns'][str(i)] = {'sdtype': 'categorical'}
    
    initial_info['column_info'] = info_copy['column_info']
    initial_info['idx_mapping'] = {str(i):i for i in list(range(len(initial_info['column_names'])))}
    initial_info['inverse_idx_mapping'] = {str(i):i for i in list(range(len(initial_info['column_names'])))}
    
    keys = list(initial_info.keys())
    for key in keys:
        if not info_copy.get(key):
            initial_info.pop(key)
    
    with open(f'./data/{name}/initial_info.json', 'w', encoding='utf-8') as f:
        json.dump(initial_info, f, ensure_ascii=False, indent=4)
    print("NEW FILE CREATED")

def postsample_OHE(dataname, path_to_save, normalizers=None):
    sample = pd.read_csv(CONFIG.get_arg('sample_save_path'))

    if normalizers:
        columns = sample.columns.copy()
        sample_q = sample[columns[range(normalizers['len_num_prev'])].tolist()].copy()
        sample = sample.drop(columns=sample.columns[range(normalizers['len_num_prev'])])
        
        sample_q = pd.DataFrame(normalizers['num_normalizer'].inverse_transform(sample_q), 
                                columns=columns[range(normalizers['len_num_prev'])]).round()
        sample = pd.DataFrame(normalizers['cat_normalizer'].inverse_transform(sample), 
                              columns=sample.columns).round()
        sample = pd.concat([sample_q, sample], axis=1)

    initial_sample = pd.DataFrame()
    
    real_data_path = f'data/{dataname}'
    initial_info_path = f'{real_data_path}/initial_info.json'
    info_path = f'{real_data_path}/info.json'
    
    with open(initial_info_path, 'r') as f:
        initial_info = json.load(f)
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    initial_num_col = [col for col in sample.columns if '_' not in col]
    initial_sample = pd.concat([initial_sample, sample[initial_num_col]])
    sample = sample.drop(columns=initial_num_col)
    
    for i in range(1, len(initial_info['column_names']) - 1): # ПОМЕНЯТЬ, ЕСЛИ ТАРГЕТОВ БОЛЬШЕ ЧЕМ 2
        cols = [col for col in sample.columns if f'{i}_' == col[:len(str(i))+1]]
        if cols:
            result = sample[cols].idxmax(axis=1)
            result = result.apply(lambda x: x.replace(f'{i}_', ''))
            initial_sample = pd.concat([initial_sample, result], axis=1)
            sample = sample.drop(columns=cols)
    
    sample[sample.shape[1]] = 1 - sample.sum(axis=1)
    sample.columns = info['initial_column_names'][-2:] # ПОМЕНЯТЬ, ЕСЛИ ТАРГЕТОВ БОЛЬШЕ ЧЕМ 2
    for i in range(1, len(initial_info['column_names'])):
        cols = [col for col in sample.columns if f'{i}_' == col[:len(str(i))+1]]
        if cols:
            result = sample[cols].idxmax(axis=1)
            result = result.apply(lambda x: x.replace(f'{i}_', ''))
            initial_sample = pd.concat([initial_sample, result], axis=1)
    initial_sample.columns = initial_info['column_names']
    initial_sample.to_csv(path_to_save, index = False)
    
    CONFIG.add_arg('sample_save_path', path_to_save)
    CONFIG.add_arg('save_path', path_to_save)
    print(f"Sample path changed to {path_to_save}")

# UTILS TRAIN
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)

# REORDER
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

# SIGMA SCHEDULER
def sigma_scheduller(name, value):
    if name == 'constant':
        return value