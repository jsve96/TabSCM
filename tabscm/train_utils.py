import networkx as nx
import pandas as pd
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq
from causallearn.search.ScoreBased.GES import ges
from sklearn.preprocessing import LabelEncoder
import random
from tabscm.notears.notears import notears




def graph_subgraph(cpdag):
    G = nx.DiGraph()
    ### get subgraph only with directed edges
    ind_un = []
    try:
        for i in range(cpdag.graph.shape[0]):
            for j in range(cpdag.graph.shape[1]):
                if cpdag.graph[i, j] == 1:  # directed edge i → j
                    G.add_edge(j, i)
                elif cpdag.graph[i,j] == -1 and cpdag.graph[j,i] == -1:
                    ind_un.append((i,j))
    except:
        print(type(cpdag))
        G  = cpdag


    ##break cycles
    try:
        while True:
            cycle = list(nx.find_cycle(G, orientation='original'))
            # Randomly choose an edge in the cycle to remove
            edge_to_remove = random.choice(cycle)
            G.remove_edge(edge_to_remove[0], edge_to_remove[1])
    except nx.exception.NetworkXNoCycle:
        pass  # No more cycles

    ## add directions for undirected edges according to top ordering
    if ind_un == []:
        return G
    unique_nodes = list({tuple(sorted(t)) for t in ind_un})

    for node in list(nx.topological_sort(G)):
        for tup in unique_nodes:
            if node in tup:
                print('add edge')
                if node == tup[0]:

                    G.add_edge(tup[0],tup[1])
                else:
                    G.add_edge(tup[1],tup[0])
                unique_nodes.pop()
    try:
        while True:
            cycle = list(nx.find_cycle(G, orientation='original'))
                # Randomly choose an edge in the cycle to remove
            edge_to_remove = random.choice(cycle)
            G.remove_edge(edge_to_remove[0], edge_to_remove[1])
    except nx.exception.NetworkXNoCycle:
            pass  # No more cycles
    

    return G  



def check_missing_node(dag,df):
    features = np.arange(df.shape[1])
    nodes = dag.nodes
    
    add_roots = [x for x in features if x not in nodes]

    if add_roots == []:
        return dag
    else:
        for node in add_roots:
            dag.add_node(node)
    return dag


def infer_casual_graph(data,method,alpha=0.01,test=fisherz):

    if method =='pc':
        cg = pc(data,alpha=alpha,ci_test=test, verbose=False)
        return cg.G
    elif method == 'ges':
        Record = ges(data)
        return Record['G']
    elif method == 'notears':
        print('start infer')
        dag_notears = notears(data,lambda1=0.01,loss_type='l2',w_threshold=0.01)
        #dag_notears = notears(data,lambda1=0.01,loss_type='l2',w_threshold=0.1)
        print('finished infer')
        DAG_NOTEAR = nx.DiGraph()
        num_nodes = dag_notears.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = dag_notears[i, j]
                if weight != 0:
                    DAG_NOTEAR.add_edge(i, j, weight=weight)
        return DAG_NOTEAR

    else:
        raise ValueError(f'Methof: {method} not implemented use pc or ges')



def process(df,info,name):

    data = df.copy()
    cols_to_encode = info['cat_col_idx'].copy()

    if info['task_type'] != 'regression':
        cols_to_encode.append(info['target_col_idx'][0])
    

    # Dictionary to hold encoders
    if name == 'adult':
        #### education.num removed from train_df
        inv_map_eduction_num = df[['education','education.num']].groupby(['education']).apply(lambda x: x['education.num'].unique()).astype(int)
        df = df.drop(columns=['education.num'])

        encoders = {}
        encoded_cols = []
        # Encode each column
        for col in cols_to_encode:
           # print(col)
            le = LabelEncoder()
            df.iloc[:,col] = le.fit_transform(df.iloc[:,col])
            encoders[col] = le 
            encoded_cols.append(col)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f'Mapping of column col {col} with: {le_name_mapping}')
    

        N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
        info['n_classes'] = N_CLASSES

        info['col_dtype'] = {i:df.dtypes[i] for i in info['num_col_idx']}
        info['col_dtype'][0] = 'int'
        info['col_dtype'][9] = 'int'
        info['col_dtype'][10] = 'int'
        info['col_dtype'][11] = 'int'


        return df,info,encoders,encoded_cols, inv_map_eduction_num



    if name != 'beijing':
        encoders = {}
        encoded_cols = []
        for col in cols_to_encode:
            if df.iloc[:,col].dtype == 'O':
                le = LabelEncoder()
                df.iloc[:,col] = le.fit_transform(df.iloc[:,col])
                
                encoders[col] = le 
                encoded_cols.append(col)

    if name == 'heloc':
        N_CLASSES = {i: int(df[c].max()+1) for i,c in enumerate(df.columns)}
        info['col_dtype'] = {i:'int' for i in info['num_col_idx']}
        info['n_classes'] = N_CLASSES
        info['col_dtype'][0] = 'binary'


    if name == 'magic':
        
        info['col_dtype'] = {i: df.dtypes[i] for i in info['num_col_idx']}
        info['n_classes'] = {i: int(df[c].max()+1) for i,c in enumerate(df.columns)}
       
        # for col in cols_to_encode:
        #     le = LabelEncoder()
        #     #print(f'DTYPE col {col} is {df.iloc[:,col].dtype}')
        #     #print(df.iloc[:,col].unique())
        #     df.iloc[:,col] = le.fit_transform(df.iloc[:,col].astype('int'))

        #     #print(type(list(le.classes_)[0]))
        #     encoders[col] = le 
        #     encoded_cols.append(col)


        
    elif name == 'loan':
        N_CLASSES = {i: int(df[c].max()+1) for i,c in enumerate(df.columns)}#{i: df[c].nunique() for i,c in enumerate(df.columns)}
        info['n_classes'] = N_CLASSES
        #info['n_classes'][1] = info['n_classes'][1] +1
        #df['Age'] = df['Age'] - df['Age'].min()
        #df['CURRENT_HOUSE_YRS'] = df['CURRENT_HOUSE_YRS'] - df['CURRENT_HOUSE_YRS'].min()
        #df['CURRENT_JOB_YRS'] = df['CURRENT_JOB_YRS'] - df['CURRENT_JOB_YRS'].min()
        info['col_dtype'] = {0:'float', 1:'int'}



    elif name == 'early_diab':
        N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
        info['n_classes'] = N_CLASSES
        #df['Age'] = df['Age'] - df['Age'].min()
        info['n_classes'][0] = df.Age.max().astype('int')+1 
        info['num_col_idx'] = []
        info['cat_col_idx'].insert(0,0)
        info['col_dtype'] = {0:'int'}

    elif name == 'beijing':
        encoded_cols = []
        encoders = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            #print(f'DTYPE col {col} is {df.iloc[:,col].dtype}')
            #print(df.iloc[:,col].unique())
            df.iloc[:,col] = le.fit_transform(df.iloc[:,col])

            #print(type(list(le.classes_)[0]))
            encoders[col] = le 
            encoded_cols.append(col)
            N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
            info['n_classes'] = N_CLASSES
            info['col_dtype'] = {i: df.dtypes[i] for i in info['num_col_idx']}
            info['col_dtype'][info['target_col_idx'][0]] = df.dtypes[info['target_col_idx'][0]]
            info['col_dtype'][10] = 'int'
            info['col_dtype'][11] = 'int'
            info['col_dtype'][5] = 'int'
            #print(encoded_cols)


    
    elif name =='housing':
        N_CLASSES = {i: int(df[c].max()+1) for i,c in enumerate(df.columns)}
        info['col_dtype'] = {i: df.dtypes[i] for i in info['num_col_idx']}
        info['n_classes'] = N_CLASSES
        info['col_dtype'][info['target_col_idx'][0]] = df.dtypes[info['target_col_idx'][0]]



    return df,info,encoders,encoded_cols




import os
import json
import pickle
import joblib

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_scm(scm: dict, save_dir: str = "saved_scm", experiment: str = "default"):
    os.makedirs(save_dir, exist_ok=True)
    scm_meta = {}

    for node, model_entry in scm.items():
        node_dir = os.path.join(save_dir, f"{experiment}_node_{node}")
        os.makedirs(node_dir, exist_ok=True)

        # Handle root node sampler (no parents)
        if not isinstance(model_entry, tuple):
            with open(os.path.join(node_dir, "sampler.pkl"), "wb") as f:
                pickle.dump(model_entry, f)
            scm_meta[node] = {
                "model_type": "sampler",
                "parents": [],
                "model_path": os.path.join(f"{experiment}_node_{node}", "sampler.pkl"),
                "n_classes": None,
            }
            continue

        model, parents, _, n_classes = model_entry

        if hasattr(model, "save") and callable(getattr(model, "save")):
            # Save DiffusionRegressor
            model.save(node_dir, node)
            model_type = "diffusion"
        else:
            # Save XGBoost model
            model.save_model(os.path.join(node_dir, "xgb_model.json"))
            model_type = "xgb"

        scm_meta[node] = {
            "model_type": model_type,
            "parents": parents,
            "model_path": node_dir,
            "n_classes": n_classes,
        }

    # Save metadata
    scm_meta = {int(k): v for k,v in scm_meta.items()}
    with open(os.path.join(save_dir, f"{experiment}_scm_meta.json"), "w") as f:
        json.dump(scm_meta, f, indent=4,cls=NpEncoder)