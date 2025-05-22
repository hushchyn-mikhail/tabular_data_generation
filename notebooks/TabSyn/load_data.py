
import os
from urllib import request
import numpy as np
import pandas as pd
import zipfile
import json

class Load_Process_Dataset():

    def __init__(self, name):
        super(Load_Process_Dataset, self).__init__()
        self.name = name

    def download_from_uci(self):
        NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'magic': 'https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip',
    'shoppers': 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
    'beijing': 'https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip',
    'news': 'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip'
}
        print(f'Start processing dataset {self.name} from UCI.')
        save_dir = f'data/{self.name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


            url = NAME_URL_DICT_UCI[self.name]
            request.urlretrieve(url, f'{save_dir}/{self.name}.zip')
            print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')

            with zipfile.ZipFile(f'{save_dir}/{self.name}.zip', 'r') as zip_ref:
                zip_ref.extractall(save_dir)

            print(f'Finish unzipping {self.name}.')

        else:
            print('Aready downloaded.')

    def get_column_name_mapping(self, data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):

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


    def train_val_test_split(self, data_df, cat_columns, num_train = 0, num_test = 0):
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
    def preprocess_beijing(self):
        with open(f'data/Info/beijing.json', 'r') as f:
            info = json.load(f)
        
        data_path = info['raw_data_path']

        data_df = pd.read_csv(data_path)
        columns = data_df.columns

        data_df = data_df[columns[1:]]


        df_cleaned = data_df.dropna()
        df_cleaned.to_csv(info['data_path'], index = False)

    def process_data(self):
        if self.name == 'beijing':
            self.preprocess_beijing()
        with open(f'data/Info/{self.name}.json', 'r') as f:
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

        idx_mapping, inverse_idx_mapping, idx_name_mapping = self.get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

        num_columns = [column_names[i] for i in num_col_idx]
        cat_columns = [column_names[i] for i in cat_col_idx]
        target_columns = [column_names[i] for i in target_col_idx]

        # if testing data is given
        if info['test_path']:

                # if testing data is given
                test_path = info['test_path']

                with open(test_path, 'r') as f:
                    lines = f.readlines()[1:]
                    test_save_path = f'data/{self.name}/test.data'
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

            train_df, test_df, seed = self.train_val_test_split(data_df, cat_columns, num_train, num_test)


        train_df.columns = range(len(train_df.columns))
        test_df.columns = range(len(test_df.columns))

        print(self.name, train_df.shape, test_df.shape, data_df.shape)

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


        save_dir = f'data/{self.name}'
        np.save(f'{save_dir}/X_num_train.npy', X_num_train)
        np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
        np.save(f'{save_dir}/y_train.npy', y_train)

        np.save(f'{save_dir}/X_num_test.npy', X_num_test)
        np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
        np.save(f'{save_dir}/y_test.npy', y_test)

        train_df[num_columns] = train_df[num_columns].astype(np.float32)
        test_df[num_columns] = test_df[num_columns].astype(np.float32)


        train_df.to_csv(f'{save_dir}/train.csv', index = False)
        test_df.to_csv(f'{save_dir}/test.csv', index = False)

        if not os.path.exists(f'synthetic/{self.name}'):
            os.makedirs(f'synthetic/{self.name}')

        train_df.to_csv(f'synthetic/{self.name}/real.csv', index = False)
        test_df.to_csv(f'synthetic/{self.name}/test.csv', index = False)

        print('Numerical', X_num_train.shape)
        print('Categorical', X_cat_train.shape)

        info['column_names'] = column_names
        info['train_num'] = train_df.shape[0]
        info['test_num'] = test_df.shape[0]

        info['idx_mapping'] = idx_mapping
        info['inverse_idx_mapping'] = inverse_idx_mapping
        info['idx_name_mapping'] = idx_name_mapping

        metadata = {'columns': {}}
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

        if info['task_type'] == 'regression':
        
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

        print(f'Processing and Saving {self.name} Successfully!')

        print(self.name)
        print('Total', info['train_num'] + info['test_num'])
        print('Train', info['train_num'])
        print('Test', info['test_num'])
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
        print('Num', num)
        print('Cat', cat)

    def start_load(self):
        self.download_from_uci()
        self.process_data()