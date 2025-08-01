import os
import torch
import argparse
import warnings
import time
from tqdm import tqdm
from tabscm.model import *
import pandas as pd
from tabscm.train_utils import *
import json
from tabscm.sample_utils import *



def main(args):

    dataname = args.dataname
 
    dataset_path = f'data/{dataname}/train.csv'
    curr_dir = os.path.dirname(os.path.abspath(__file__))


    try:
        with open(f"{curr_dir}/Info/{dataname}/info.json") as f:
            info = json.load(f)
    except:
         with open(f"data/{dataname}/info.json") as f:
            info = json.load(f)

    train_df = pd.read_csv(dataset_path)
    if dataname == 'adult':
        _,info,encoders,encoded_cols,inv_map = process(df=train_df,info = info,name=dataname)
    else:

        _,info,encoders,encoded_cols = process(df=train_df,info = info,name=dataname)
    print(info)
    ### load scm model
    #loaded_scm = load_tabscm()

    with open(f'{curr_dir}/models/{dataname}/dag/dag.json') as f:
        data_dag = json.load(f)
    loaded_dag = generate_dag_from_dict(data_dag)
    print(loaded_dag.nodes)

    exp_save =  f'{curr_dir}/models/{dataname}'
    loaded_scm = load_scm(f'{exp_save}/scm',device='cuda')
    #print(loaded_scm)
    
    
    n_samples = info['train_num']
    ### sample from scm model
    with open(args.intervention_config) as f:
        config_inter = json.load(f)
    intervention = {int(k): v for k,v in config_inter.items()}
    print(intervention)
    for node in intervention.keys():
        if node in encoded_cols:
            print('HERE')
            encoded_val = encoders[int(node)].transform(np.array(intervention[int(node)]).reshape(-1,1).ravel())
            intervention[int(node)] = encoded_val
    
    samples = sample_interventions_from_scm(loaded_scm,loaded_dag,n_samples,intervention,info)

    save_path = args.save_path

    #### need to postprocess samples ---> convert with encoder
    samples_df = pd.DataFrame(samples,columns=info['column_names'])
    
    convert_dict = {train_df.columns[col]: train_df.dtypes[col] for col in range(train_df.shape[1])}

    for col in encoded_cols:
        if col in encoders.keys():
            print(f'Column: {col}')
            
            samples_df.iloc[:,col] = encoders[col].inverse_transform(samples_df.iloc[:,col].astype('int'))


    if dataname == 'adult':
        vals = samples_df['education'].apply(lambda x: inv_map[inv_map.index == x].values[0]).values
        samples_df.insert(4,'education.num',vals)


    samples_df = samples_df.astype(convert_dict)
    samples_df.to_csv(save_path, index = False)
    print(samples_df.head())


    print('Saving sampled data to {}'.format(save_path))


