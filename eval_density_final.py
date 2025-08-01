import numpy as np
import pandas as pd
import os 

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import LogisticDetection



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx
    print(num_col_idx)
    print(cat_col_idx)

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    columns = { int(k):item for k,item in columns.items()}
    #print(columns)
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        #print(len(num_col_idx))
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_paths = [f'synthetic/{dataname}/{model}.csv']
    elif os.path.isdir(args.path):
        syn_paths = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.csv')]
    else:
        syn_paths = [args.path]

    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}' 

    # with open(f'{data_dir}/info.json', 'r') as f:
    #     info = json.load(f)
    #print(info)
    real_data = pd.read_csv(real_path)
    real_data.columns = range(len(real_data.columns))

    all_results = []
    C2ST = []
    for syn_path in syn_paths:
        print(syn_path)
        with open(f'{data_dir}/info.json', 'r') as f:
            info = json.load(f)
        syn_data = pd.read_csv(syn_path)
        syn_data.columns = range(len(syn_data.columns))
        #print(syn_data.columns)

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)

        diag_report = DiagnosticReport()
        diag_report.generate(new_real_data, new_syn_data, metadata)

        quality = qual_report.get_properties()
        diag = diag_report.get_properties()

        Shape = quality['Score'][0]
        Trend = quality['Score'][1]
        Quality = (Shape + Trend) / 2

        syn_file_name = os.path.basename(syn_path).replace('.csv', '')

        all_results.append({
            'file': syn_file_name,
            'Shape': Shape,
            'Trend': Trend,
            'Quality': Quality
        })

    #     score = LogisticDetection.compute(
    #     real_data=new_real_data,
    #     synthetic_data=new_syn_data,
    #     metadata=metadata
    # )
    #    # C2ST.append(score)


        # # Optional: Save individual shape/trend/coverage reports per file
        # save_dir = f'eval/density/{dataname}/{model}/{syn_file_name}'
        # os.makedirs(save_dir, exist_ok=True)

        # shapes = qual_report.get_details(property_name='Column Shapes')
        # trends = qual_report.get_details(property_name='Column Pair Trends')
        # coverages = diag_report.get_details('Data Structure')

        # shapes.to_csv(f'{save_dir}/shape.csv')
        # trends.to_csv(f'{save_dir}/trend.csv')
        # coverages.to_csv(f'{save_dir}/coverage.csv')

    # Save aggregated quality metrics
    results_df = pd.DataFrame(all_results)
    #m = np.mean(np.array(C2ST))
    #s = np.std(np.array(C2ST))
    #print(f'CS2T :{m} +- {s}')
    save_dir = f'syn_data_eval/density/{dataname}/{model}'
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(f'{save_dir}/aggregated_quality.csv', index=False)

    print("Evaluation complete. Aggregated results saved to:")
    print(f"{save_dir}/aggregated_quality.csv")
    print(f'Column Fit error {(1-results_df.Shape.mean())*100}% +- {results_df.Shape.std()*100}')
    print(f'Correlation Fit error {(1-results_df.Trend.mean())*100}% +- {results_df.Trend.std()*100}')