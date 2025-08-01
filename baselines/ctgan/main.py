import argparse
import warnings
import time
from ctgan import CTGAN
import os, json
import pandas as pd
import numpy as np


def main(args): 

    dataname = args.dataname
    dataset_path = f'data/{dataname}/train.csv'
    train_df = pd.read_csv(dataset_path)
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    with open(f"data/{dataname}/info.json") as f:
            info = json.load(f)


    task_type = info['task_type']

    if task_type == 'regression':
         
        discrete_columns_idx =  info['cat_col_idx']

    else:
         
         discrete_columns_idx = info['cat_col_idx']
         discrete_columns_idx.append(info['target_col_idx'][0])
    print(discrete_columns_idx)
    discrete_columns = [info['column_names'][i] for i in discrete_columns_idx]




    ckpt_dir = f'{curr_dir}/ckpt'

    if not os.path.exists(ckpt_dir):
         os.makedirs(ckpt_dir)

    ckpt_path = f'{ckpt_dir}/{dataname}'

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)


    start_time = time.time()


    ctgan = CTGAN(epochs=300,verbose=True)
    ctgan.fit(train_df,discrete_columns)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    ctgan.save(f'{ckpt_path}/model.pt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of CTGAN')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    #parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

