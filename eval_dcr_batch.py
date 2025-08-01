import numpy as np
import torch
import pandas as pd
import json
import os
import sys
import argparse
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_train import preprocess, TabularDataset  # assuming used elsewhere

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default=None, help='Path to synthetic .csv or directory of .csv files')
args = parser.parse_args()

# def compute_dcr(syn_data_np, real_data_th, test_data_th, batch_size=100):
#     syn_data_th = torch.tensor(syn_data_np).to(real_data_th.device)
#     dcrs_real = []
#     dcrs_test = []

#     for i in range((syn_data_th.shape[0] // batch_size) + 1):
#         batch = syn_data_th[i * batch_size:(i + 1) * batch_size]
#         if batch.shape[0] == 0:
#             continue
#         dcr_real = (batch[:, None] - real_data_th).abs().sum(dim=2).min(dim=1).values
#         dcr_test = (batch[:, None] - test_data_th).abs().sum(dim=2).min(dim=1).values
#         dcrs_real.append(dcr_real)
#         dcrs_test.append(dcr_test)

#     dcrs_real = torch.cat(dcrs_real)
#     dcrs_test = torch.cat(dcrs_test)
#     score = (dcrs_real < dcrs_test).float().mean().item()
#     avg_dcr_real = dcrs_real.mean().item()

#     return score, avg_dcr_real


def compute_dcr(syn_data_np, real_data_th, test_data_th, batch_size=100, ref_batch_size=10000):
    device = real_data_th.device
    syn_data_th = torch.tensor(syn_data_np, dtype=torch.float32).to(device)
    dcrs_real = []
    dcrs_test = []

    for i in range(0, syn_data_th.shape[0], batch_size):
        batch = syn_data_th[i:i + batch_size]  # [B, F]
        if batch.shape[0] == 0:
            continue

        # Compute distance to real data in chunks
        dists_real = []
        for j in range(0, real_data_th.shape[0], ref_batch_size):
            ref = real_data_th[j:j + ref_batch_size]  # [R, F]
            d = torch.cdist(batch, ref, p=1)  # [B, R], L1 distance
            dists_real.append(d)
        dcr_real = torch.cat(dists_real, dim=1).min(dim=1).values  # min over R

        # Same for test data
        dists_test = []
        for j in range(0, test_data_th.shape[0], ref_batch_size):
            ref = test_data_th[j:j + ref_batch_size]
            d = torch.cdist(batch, ref, p=1)
            dists_test.append(d)
        dcr_test = torch.cat(dists_test, dim=1).min(dim=1).values

        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)

    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)
    score = (dcrs_real < dcrs_test).float().mean().item()
    avg_dcr_real = dcrs_real.mean().item()
    return score, avg_dcr_real



if __name__ == '__main__':
    dataname = args.dataname
    model = args.model
    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'
    data_dir = f'data/{dataname}'

    # Load metadata
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Load and preprocess real/test data (once)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)
    real_data.columns = list(np.arange(real_data.shape[1]))
    test_data.columns = list(np.arange(real_data.shape[1]))

    num_ranges = [(real_data[i].max() - real_data[i].min()) for i in num_col_idx]
    num_ranges = np.array(num_ranges)

    def preprocess_data(df):
        df.columns = list(np.arange(len(df.columns)))
        num = df[num_col_idx].to_numpy() / num_ranges
        cat = df[cat_col_idx].to_numpy().astype(str)
        return num, cat

    num_real, cat_real = preprocess_data(real_data)
    num_test, cat_test = preprocess_data(test_data)

    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encoder.fit(cat_real)
    cat_real_oh = encoder.transform(cat_real)#.toarray()
    cat_test_oh = encoder.transform(cat_test)#.toarray()

    real_np = np.concatenate([num_real, cat_real_oh], axis=1)
    test_np = np.concatenate([num_test, cat_test_oh], axis=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_th = torch.tensor(real_np, dtype=torch.float32).to(device)
    test_th = torch.tensor(test_np, dtype=torch.float32).to(device)

    # Collect multiple results
    results = []

    paths = []
    if not args.path:
        paths = [f'synthetic/{dataname}/{model}.csv']
    elif os.path.isdir(args.path):
        paths = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.csv')]
    else:
        paths = [args.path]

    for path in paths:
        syn_data = pd.read_csv(path)
        syn_data = syn_data.astype({col: real_data.dtypes[i] for i,col in enumerate(syn_data.columns)})
        print(syn_data.head())
        print(syn_data.dtypes)
        syn_data.columns = list(np.arange(syn_data.shape[1]))

        num_syn, cat_syn = preprocess_data(syn_data)
        cat_syn_oh = encoder.transform(cat_syn)#.toarray()
        syn_np = np.concatenate([num_syn, cat_syn_oh], axis=1)

        score, avg_dcr = compute_dcr(syn_np, real_th, test_th)
        results.append((os.path.basename(path), score, avg_dcr))
        print(f"{os.path.basename(path)}: DCR Score = {score:.4f}, Avg DCR = {avg_dcr:.4f}")

    if len(results) > 1:
        all_scores = np.array([s for _, s, _ in results])
        all_dcrs = np.array([d for _, _, d in results])
        print("\nSummary over all files:")
        print(f"Mean DCR Score = {all_scores.mean():.4f}, Std = {all_scores.std():.4f}")
        print(f"Mean Avg DCR = {all_dcrs.mean():.4f}, Std = {all_dcrs.std():.4f}")
