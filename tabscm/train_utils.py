import networkx as nx
import pandas as pd
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq
from causallearn.search.ScoreBased.GES import ges
from sklearn.preprocessing import LabelEncoder
import random




def graph_subgraph(cpdag):
    G = nx.DiGraph()
    ### get subgraph only with directed edges
    ind_un = []
    for i in range(cpdag.graph.shape[0]):
        for j in range(cpdag.graph.shape[1]):
            if cpdag.graph[i, j] == 1:  # directed edge i → j
                G.add_edge(j, i)
            elif cpdag.graph[i,j] == -1 and cpdag.graph[j,i] == -1:
                ind_un.append((i,j))


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


def infer_casual_graph(data,method,alpha=0.01):

    if method =='pc':
        cg = pc(data,alpha=0.01,ci_test=fisherz, verbose=False)
        return cg
    elif method == 'ges':
        Record = ges(data)
        return Record['G']
    else:
        raise ValueError(f'Methof: {method} not implemented use pc or ges')



def process(df,info,name):

    data = df.copy()
    cols_to_encode = info['cat_col_idx'].copy()
    if info['task_type'] != 'regression':
        cols_to_encode.append(info['target_col_idx'][0])
    

    # Dictionary to hold encoders
    encoders = {}
    cols_to_encode
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
        
    elif name == 'loan':
        N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
        info['n_classes'] = N_CLASSES
        info['n_classes'][1] = info['n_classes'][1] +1
        df['Age'] = df['Age'] - df['Age'].min()
        df['CURRENT_HOUSE_YRS'] = df['CURRENT_HOUSE_YRS'] - df['CURRENT_HOUSE_YRS'].min()
        df['CURRENT_JOB_YRS'] = df['CURRENT_JOB_YRS'] - df['CURRENT_JOB_YRS'].min()
        info['col_dtype'] = {0:'float', 1:'int'}



    elif name == 'early_diab':
        N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
        info['n_classes'] = N_CLASSES
        df['Age'] = df['Age'] - df['Age'].min()
        info['n_classes'][0] = df.Age.max().astype('int')+1 
        info['num_col_idx'] = []
        info['cat_col_idx'].insert(0,0)

    elif name == 'beijing':

        for col in cols_to_encode:
            le = LabelEncoder()
            df.iloc[:,col] = le.fit_transform(df.iloc[:,col])
            encoders[col] = le  
            N_CLASSES = {i: df[c].nunique() for i,c in enumerate(df.columns)}
            info['n_classes'] = N_CLASSES
            info['col_dtype'] = {i: df.dtypes[i] for i in info['num_col_idx']}

    elif name == 'adult':
        return None








    



    return data,info,encoders,encoded_cols
