import sys
sys.path.append('../..')
from utils import *

import os
import argparse
import torch
import numpy as np
import pandas as pd
import time
import json

import src
from models.tabddpm.train import train
from models.tabddpm.sample import sample
from models.tabddpm.modules import GaussianMultinomialDiffusion, MLPDiffusion

class TabDDPM():
    def __init__(self, CONFIG, model_save_path=None):
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG = CONFIG
        self.dataname = self.CONFIG.get_arg('dataname')
        self.device = self.CONFIG.get_arg('device')

        self.config_path = f'{self.curr_dir}/configs/{self.dataname}.toml'
        if model_save_path:
            self.model_save_path = model_save_path
        else:
            self.model_save_path = f'{self.curr_dir}/ckpt/{self.dataname}'
        self.real_data_path = f'data/{self.dataname}'

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        
        self.CONFIG.add_arg('train', 1)
        self.raw_config = src.load_config(self.config_path)

    def train(self):
        ''' 
        Modification of configs
        '''
        print('START TRAINING')
    
        train(
            **self.raw_config['train']['main'],
            **self.raw_config['diffusion_params'],
            model_save_path=self.model_save_path,
            real_data_path=self.real_data_path,
            task_type=self.raw_config['task_type'],
            model_type=self.raw_config['model_type'],
            model_params=self.raw_config['model_params'],
            T_dict=self.raw_config['train']['T'],
            num_numerical_features=self.raw_config['num_numerical_features'],
            device=self.device
        )

    def sample( 
            self,
            sample_save_path,
            ddim=False,
            steps=1000
      ):
        sample(
            num_samples=self.raw_config['sample']['num_samples'],
            batch_size=self.raw_config['sample']['batch_size'],
            disbalance=self.raw_config['sample'].get('disbalance', None),
            **self.raw_config['diffusion_params'],
            model_save_path=self.model_save_path,
            sample_save_path=sample_save_path,
            real_data_path=self.real_data_path,
            task_type=self.raw_config['task_type'],
            model_type=self.raw_config['model_type'],
            model_params=self.raw_config['model_params'],
            T_dict=self.raw_config['train']['T'],
            num_numerical_features=self.raw_config['num_numerical_features'],
            device=self.device,
            ddim=ddim,
            steps=steps
      )