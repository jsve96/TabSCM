import os
import torch
import argparse
import warnings
import time
from tqdm import tqdm
from model import *
import pandas as pd
from train_utils import *
import json

warnings.filterwarnings('ignore')



def main(args): 
    device = args.device
    dataname = args.dataname
    dataset_path = f'data/{dataname}/train.csv'
    train_df = pd.read_csv(dataset_path)
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        with open(f"{curr_dir}/Info/{dataname}/info.json") as f:
            INFO = json.load(f)
    except:
         with open(f"data/{dataname}/info.json") as f:
            INFO = json.load(f)
    

    data = process(data=train_df,info = INFO,name=dataname)

    start_time = time.time()

    end_time = time.time()
    print('Time: ', end_time - start_time)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Training of TabSCM')

#     parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
#     parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

#     args = parser.parse_args()

#     # check cuda
#     if args.gpu != -1 and torch.cuda.is_available():
#         args.device = f'cuda:{args.gpu}'
#     else:
#         args.device = 'cpu'