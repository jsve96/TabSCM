import os
import torch
import argparse
import warnings
import time
from tqdm import tqdm
from tabscm.model import *
import pandas as pd
from tabscm.train_utils import *
from tabscm.sample_utils import*
import json
import shutil
from causallearn.utils.cit import fisherz, chisq
from causallearn.graph.GeneralGraph import GeneralGraph


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
            print('alternative info')
    except:
         with open(f"data/{dataname}/info.json") as f:
            INFO = json.load(f)

    
    if dataname == 'adult':
        df,info,encoders,encoded_cols,_ = process(df=train_df,info = INFO,name=dataname)
    else: 
        df,info,encoders,encoded_cols = process(df=train_df,info = INFO,name=dataname)
    data = df.to_numpy().astype('float')
    print(info)
    
    if args.version == 'medium':
        if dataname == 'heloc':
            params_regressor = {
                'timesteps' : 500,
                'epochs' : 1000
            }
        elif dataname == 'loan':
             params_regressor = {
                'timesteps' : 1500,
                'epochs' : 500
            }
        elif dataname == 'magic':
            params_regressor = {
                'timesteps' : 2000,#2000,#1500,
                'epochs' : 2000#200#500
            }
        elif dataname == 'housing':
            params_regressor = {
                'timesteps' : 1000,
                'epochs' : 500 #vorher 1000
            }
        elif dataname == 'beijing':
            params_regressor = {
                'timesteps' : 500, #vorher 1500
                'epochs' : 500 #vorher 500
            } 
            # notears w=0.1
        else:
            params_regressor = {
                'timesteps' : 500,
                'epochs' : 500
            }

    alpha = 0.01 if dataname!='loan' else 0.001
    #alpha = 0.05
    alpha =0.05
    alpha = 0.1 #magic

    start_time = time.time()

    if args.ci_test == 'fisherz':
        test = fisherz
    else:
        alpha =0.1
        test = chisq
    con =False
    if con:
        cg = infer_casual_graph(data,method=args.cd_alg,alpha=alpha,test=test)
        print(cg)
        if args.cd_alg == 'notears':
            final_dags = [cg]
            print(cg)
            final_dags = [graph_subgraph(dag) for dag in final_dags]
            final_dags = [check_missing_node(dag,df) for dag in final_dags]
        else:
            sampled_dags = [graph_subgraph(cg) for _ in range(1)]
            final_dags = [check_missing_node(dag,df) for dag in sampled_dags]
            if args.dataname == 'housing':
                if (7,6) not in final_dags[0].edges and (6,7) not in final_dags[0].edges:
                    print('add edge')
                    final_dags[0].add_edge((6,7))

    ### comment out later
    with open(f'{curr_dir}/models/{dataname}/dag/dag.json') as f:
        data_dag = json.load(f)
    final_dags = [generate_dag_from_dict(data_dag)]

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
    save_dag['alg'] = args.cd_alg
    save_dag['ci_test'] = args.ci_test
    save_dag['alpha'] = alpha
    with open(f'{exp_save}/dag/dag.json',"w") as f:
        json.dump(save_dag,f,indent=4,cls=NpEncoder)


    print(scm)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSCM')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    #parser.add_argument('--ci_test', type=str, default='fisherz', help='GPU index.')
    


    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'