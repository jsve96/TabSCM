import os
import glob
import pandas as pd
import argparse
from eval.mle.mle import compute_scores, get_evaluator
import numpy as np
import json

def load_csv(path):
    return pd.read_csv(path).values

def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def main(args):

    syn_files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.csv')]
    
    # Load real train/test data & metadata
    if args.train is None:
        train = load_csv(f'./synthetic/{args.dataname}/real.csv')
    else:
        train = [load_csv(args.train)]
    if args.test is None:
        test = load_csv(f'./synthetic/{args.dataname}/test.csv')
    else:
        test = load_csv(args.test)
    
    info = load_metadata(f'./data/{args.dataname}/info.json')
    #print(info)

    task_type = info['task_type']

    evaluator = get_evaluator(task_type)


    all_results = []

    for i, syn_file in enumerate(syn_files):
        print(f"Evaluating {syn_file}...")
       
        train = load_csv(syn_file)

        if task_type == 'regression':
            best_r2_scores, best_rmse_scores = evaluator(train, test, info)
            
            overall_scores = {}
            for score_name in ['best_r2_scores', 'best_rmse_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 

        else:

            best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)

            overall_scores = {}
            for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 

        all_results.append(overall_scores)
    flat_results = []
        #all_results.append(result)

    for res in all_results:
        flat = {}
        for key, value in res.items():  # key = 'best_f1_scores', etc.
            for model_name, metrics in value.items():  # model_name = 'XGBClassifier'
                for metric_name, score in metrics.items():  # metric_name = 'accuracy', etc.
                    flat_key = f"{key}.{model_name}.{metric_name}"  # e.g. 'best_f1_scores.XGBClassifier.accuracy'
                    flat[flat_key] = float(score)  # Ensure it's a float, not np.float64
        flat_results.append(flat)


    result_df = pd.DataFrame(flat_results)

    mean_scores = result_df.mean()
    std_scores = result_df.std()

    summary = pd.concat([mean_scores, std_scores], axis=1)
    summary.columns = ['mean', 'std']
    print('##### SUMARY #####')
    print(summary)


    # if not os.path.exists(f'./syn_data_eval'):
    # # Create a new directory because it does not exist
    #     os.makedirs(f'./syn_data_eval/mle')

    # elif not os.path.exists(f'./syn_data_eval/mle/{args.dataname}'):
    #     os.makedirs(f'./syn_data_eval/mle/{args.dataname}')
    
    # if not os.path.exists(f'./syn_data_eval/mle/{args.dataname}/{args.model}'):
    #     os.makedirs(f'./syn_data_eval/mle/{args.dataname}/{args.model}')
       




    if args.save_path is None:
        result_df.to_csv(f'./syn_data_eval/mle/{args.dataname}/{args.model}/result.csv', index=False)
    else:
        result_df.to_csv(args.save_path, index=False)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False,default=None, help='Path to real train CSV')
    parser.add_argument('--test', required=False,default=None, help='Path to real test CSV')
    parser.add_argument('--metadata', required=False, help='Path to metadata JSON')
    parser.add_argument('--path', required=True, type=str, help='Path to directory with synthetic data')
    parser.add_argument('--save_path', default=None, help='Output CSV path')
    parser.add_argument('--model',type=str)
    parser.add_argument('--dataname', type=str, required=True)
    args = parser.parse_args()

    main(args)
