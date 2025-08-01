import argparse
import warnings
import time
from ctgan import CTGAN
import os, json
import pandas as pd
import numpy as np



def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    if args.save_path is None:
         save_path = f'synthetic/{dataname}/ctgan.csv'
    else:
        save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    with open(f"data/{dataname}/info.json") as f:
            info = json.load(f)

    SIZE = info['train_num']

    ckpt_path = f'{curr_dir}/ckpt/{dataname}/model.pt'
    print(ckpt_path)

    try:
        ctgan = CTGAN.load(ckpt_path)
    except:
        raise ValueError(f'no ckpts available for {dataname}')
    
    '''
        Generating samples    
    '''
    start_time = time.time()

    syn_df = ctgan.sample(SIZE)

    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

