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
import shutil

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
    

    df,info,encoders,encoded_cols = process(df=train_df,info = INFO,name=dataname)
    data = df.to_numpy().astype('float')
    
    if args.version == 'medium':
        params_regressor = {
            'timesteps' : 500,
            'epochs' : 500
        }

    start_time = time.time()

    cg = infer_casual_graph(data,method=args.cd_alg)
    sampled_dags = [graph_subgraph(cg) for _ in range(1)]
    final_dags = [check_missing_node(dag,df) for dag in sampled_dags]
    scm = fit_scm_from_dag(data,final_dags[0],info,args.device,**params_regressor)


    end_time = time.time()
    print('Time: ', end_time - start_time)

    model_save= f'{curr_dir}/models'
    if not os.path.exists(model_save):
        os.makedir(model_save)

    exp_save =  f'{curr_dir}/models/{dataname}'

    if not os.path.exists(exp_save):
        os.makedirs(exp_save)
        os.makedirs(f'{exp_save}/dag')


    else:
        print('remove dir')
        shutil.rmtree(exp_save)

        os.makedirs(exp_save)
        os.makedirs(f'{exp_save}/dag')
    print(scm)
    save_scm(scm,f'{exp_save}/scm')

    save_dag = {'nodes':list(final_dags[0].nodes), 'edges': list(final_dags[0].edges)}
    with open(f'{exp_save}/dag/dag.json',"w") as f:
        json.dump(save_dag,f,indent=4)


    print(scm)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSCM')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'