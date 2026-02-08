import os
import torch
from tqdm import tqdm
import argparse
import numpy as np
import numpy as np 
import pandas as pd
import networkx as nx 
#from model.diffusion import create_model_from_graph
import dowhy.gcm as cy
from dowhy.gcm  import draw_samples, interventional_samples, counterfactual_samples
import json
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

from baselines.dcm.model.diffussion_modified import create_model_from_graph

warnings.filterwarnings('ignore')


def generate_dag_from_dict(data):
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node)
    for i,j in data['edges']:
        G.add_edge(i,j)

    return G



def main(args): 
    device = args.device

    dataname = args.dataname
 
    dataset_path = f'data/{dataname}/train.csv'
    curr_dir = os.path.dirname(os.path.abspath(__file__))

   

    factuals = pd.read_csv(dataset_path)

    with open(f"data/{dataname}/info.json") as f:
            info = json.load(f)

    cols_to_encode = info['cat_col_idx'].copy()

    if info['task_type'] != 'regression':
        cols_to_encode.append(info['target_col_idx'][0])

    encoder = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        factuals.iloc[:,col] = le.fit_transform(factuals.iloc[:,col])
        encoder[col] = le

    file_path = f"{curr_dir}/ckpts/{dataname}_dcm_model.pkl"
    with open(file_path, "rb") as f:

        loaded_dcm = pickle.load(f)


    obs_samples = draw_samples(loaded_dcm, num_samples = factuals.shape[0])

    df = obs_samples.reindex(sorted(obs_samples.columns), axis=1)

    for col in cols_to_encode:
        
        df[col] = encoder[col].inverse_transform(df[col].values)

    syn_columns = {y: x for x, y in {k:i for i,k in enumerate(factuals.columns)}.items()}

    syn_output = df.rename(columns=syn_columns)

    syn_output.to_csv(args.save_path,index=False)
    

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Training of DCM')

#     parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
#     parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
#     parser.add_argument('')

#     args = parser.parse_args()

#     # check cuda
#     if args.gpu != -1 and torch.cuda.is_available():
#         args.device = f'cuda:{args.gpu}'
#     else:
#         args.device = 'cpu'