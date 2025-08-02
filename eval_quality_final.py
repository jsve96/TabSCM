import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

# from eval_statistical import AlphaPrecision
# from data_loader import GenericDataLoader  # or wherever it's defined

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')  # Can be used as a prefix
parser.add_argument('--path', type=str, required=True, help='The directory path of synthetic data files')
args = parser.parse_args()


def preprocess_real_data(dataname):
    data_dir = f'data/{dataname}' 
    real_path = f'synthetic/{dataname}/real.csv'
    
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    real_data = pd.read_csv(real_path)
    real_data.columns = range(len(real_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_np = num_real_data.to_numpy()
    cat_real_np = cat_real_data.to_numpy().astype(str)

    encoder = OneHotEncoder()
    encoder.fit(cat_real_np)

    cat_real_oh = encoder.transform(cat_real_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_np, cat_real_oh), axis=1)).astype(float)

    return le_real_data, encoder, info, num_col_idx, cat_col_idx, cat_real_data


def preprocess_syn_data(filepath, info, encoder, num_col_idx, cat_col_idx,model_name,cat_real_data):
    syn_data = pd.read_csv(filepath)
    syn_data.columns = range(len(syn_data.columns))

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]
    print(cat_syn_data)
    types = {}
    for dt,col in zip(cat_syn_data.dtypes,cat_syn_data.columns):
        if dt == 'float64':
           types[col] = int
        else:
            types[col] = 'object'
    cat_syn_data = cat_syn_data.astype(types)
    cat_syn_data_np = cat_syn_data.to_numpy().astype(str)

    if (dataname == 'default' or dataname == 'news') and model_name == 'codi':
            cat_syn_data_np = cat_syn_data.astype('int').to_numpy().astype('str')

    elif model_name == 'great':
        if dataname == 'shoppers':
            cat_syn_data_np[:, 1] = cat_syn_data[11].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 2] = cat_syn_data[12].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 3] = cat_syn_data[13].astype('int').to_numpy().astype('str')
                
            max_data = cat_real_data[14].max()
            
            cat_syn_data.loc[cat_syn_data[14] > max_data, 14] = max_data
                # cat_syn_data[14] = cat_syn_data[14].apply(lambda x: threshold if x > max_data else x)
                
            cat_syn_data_np[:, 4] = cat_syn_data[14].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 4] = cat_syn_data[14].astype('int').to_numpy().astype('str')
            
        elif dataname in ['default', 'faults', 'beijing']:

            columns = cat_real_data.columns
            for i, col in enumerate(columns):
                if (cat_real_data[col].dtype == 'int'):

                    max_data = cat_real_data[col].max()
                    min_data = cat_real_data[col].min()

                    cat_syn_data.loc[cat_syn_data[col] > max_data, col] = max_data
                    cat_syn_data.loc[cat_syn_data[col] < min_data, col] = min_data

                    cat_syn_data_np[:, i] = cat_syn_data[col].astype('int').to_numpy().astype('str')
                        
            else:
                #print(cat_syn_data)
                cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

        else:
            cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
            #cat_syn_data_np[:,0] = cat_syn_data_np.astype('int').astype('str')


    num_syn_np = num_syn_data.to_numpy()
    cat_syn_oh = encoder.transform(cat_syn_data_np).toarray()

    le_syn_data = pd.DataFrame(np.concatenate((num_syn_np, cat_syn_oh), axis=1)).astype(float)

    return le_syn_data


if __name__ == '__main__':
    dataname = args.dataname
    path = args.path

    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")

    le_real_data, encoder, info, num_col_idx, cat_col_idx, cat_real_data = preprocess_real_data(dataname)

    alpha_list = []
    beta_list = []

    for fname in os.listdir(path):
        if not fname.endswith('.csv'):
            continue

        fpath = os.path.join(path, fname)
        print(f'Processing {fpath}...')

        le_syn_data = preprocess_syn_data(fpath, info, encoder, num_col_idx, cat_col_idx,model_name=args.model,cat_real_data=cat_real_data)

        X_syn_loader = GenericDataLoader(le_syn_data)
        X_real_loader = GenericDataLoader(le_real_data)

        evaluator = eval_statistical.AlphaPrecision()
        result = evaluator.evaluate(X_real_loader, X_syn_loader)
        result = {k: v for k, v in result.items() if 'naive' in k}

        alpha = result['delta_precision_alpha_naive']
        beta = result['delta_coverage_beta_naive']

        alpha_list.append(alpha)
        beta_list.append(beta)

        print(f'→ Alpha: {alpha:.4f}, Beta: {beta:.4f}')

    print('\n===== Final Results =====')
    print(f'Alpha Precision: mean={np.mean(alpha_list):.6f}, std={np.std(alpha_list):.6f}')
    print(f'Beta Recall:     mean={np.mean(beta_list):.6f}, std={np.std(beta_list):.6f}')
