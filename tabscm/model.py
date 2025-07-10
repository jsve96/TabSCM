from diffusion_regressor import DiffusionRegressor
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import networkx as nx





def is_categorical(node: int,INFO) -> bool:
    return node in INFO["cat_col_idx"] or node in INFO["target_col_idx"]

def get_num_classes(node:int,INFO) -> int:
    return INFO['n_classes'][node]

# -------------------------------
# STEP 1 — Fit SCM from DAG
# -------------------------------

def root_sampler_kde(values, bandwidth=0.05):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(values.reshape(-1, 1))
    def sampler(n):
        return kde.sample(n).flatten()
    return sampler


def root_sampler_categorical(values: np.ndarray):
    """
    Fit a categorical (multinomial) distribution sampler.
    """
    classes, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()

    def sampler(n):
        return np.random.choice(classes, size=n, p=probs)

    return sampler


def fit_scm_from_dag(data: np.ndarray, dag: nx.DiGraph, INFO, device: str) -> dict:
    scm = {}
    num_iter = len(dag.nodes)

    for node in tqdm(nx.topological_sort(dag), total=num_iter, desc='Fitting nodes',position=0):
        parents = list(dag.predecessors(node))
        y = data[:, node]

        if not parents:
            if is_categorical(node,INFO):
                sampler = root_sampler_categorical(y)
            else:
                sampler = root_sampler_kde(y)
            scm[node] = sampler

        else:
            X = data[:, parents]

            if is_categorical(node,INFO):
                print(node)
                n_classes = get_num_classes(node,INFO)
                print(n_classes)
                print(np.unique(y))
                dtrain = xgb.DMatrix(data[:,parents], label=y)
                params = {
                    'num_class': n_classes,
                    'objective': 'multi:softprob',
                    'eval_metric': 'aucpr',#'mlogloss',
                    'tree_method': 'hist',
                    'eta': 0.2,
                    'max_depth': 30,
                    'alpha':1.5,
                    'lambda':1.5,
                }
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=500,
                )
                scm[node] = (model, parents, None, n_classes) 
            else:
                model = DiffusionRegressor(device=device)
                model.fit(X, y)
                scm[node] = (model, parents, None, None)
    return scm
