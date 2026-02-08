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

    with open(f'tabscm/models/{dataname}/dag/dag.json') as f:
        data_dag = json.load(f)

    final_dags = [generate_dag_from_dict(data_dag)]

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

    factuals_input = factuals.rename(columns={k:i for i,k in enumerate(factuals.columns)})

    CAT_INFO = {k: int(factuals_input[k].max()+1) for k in cols_to_encode}

    params = {'num_epochs' : 500,
          'lr' : 1e-4,
          'batch_size': 128,
          'hidden_dim' : 64,
          'category_sizes': CAT_INFO,
          'use_positional_encoding': False}


    diff_model = create_model_from_graph(final_dags[0], params,categorical_nodes=[k for k in CAT_INFO.keys()])
    cy.fit(diff_model, factuals_input)

    file_path = f"{curr_dir}/ckpts/{dataname}_dcm_model.pkl"
    # 1. Save the model
    with open(file_path, "wb") as f:
        pickle.dump(diff_model, f)


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of DCM')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'