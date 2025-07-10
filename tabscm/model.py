from tabscm.diffusion_regressor import DiffusionRegressor
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import networkx as nx



def is_categorical(node: int,INFO) -> bool:
    return node in INFO["cat_col_idx"] or node in INFO["target_col_idx"]

def get_num_classes(node:int,INFO) -> int:
    return INFO['n_classes'][node]




#  — Fit SCM from DAG


def root_sampler_kde(values, bandwidth=0.05):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(values.reshape(-1, 1))
    def sampler(n):
        return kde.sample(n).flatten()
    return sampler


# def root_sampler_categorical(values: np.ndarray):
#     """
#     Fit a categorical (multinomial) distribution sampler.
#     """
#     classes, counts = np.unique(values, return_counts=True)
#     probs = counts / counts.sum()

#     def sampler(n):
#         return np.random.choice(classes, size=n, p=probs)

#     return sampler



class RootSamplerCategorical:
    def __init__(self, y):
        # Fit a histogram or empirical distribution
        values, counts = np.unique(y, return_counts=True)
        self.values = values
        self.probs = counts / counts.sum()

    def __call__(self, n):
        return np.random.choice(self.values, size=n, p=self.probs)

def fit_scm_from_dag(data: np.ndarray, dag: nx.DiGraph, INFO, device: str,**params_regressor) -> dict:
    scm = {}
    num_iter = len(dag.nodes)

    for node in tqdm(nx.topological_sort(dag), total=num_iter, desc='Fitting nodes',position=0):
        parents = list(dag.predecessors(node))
        y = data[:, node]

        if not parents:
            if is_categorical(node,INFO):
                #sampler = root_sampler_categorical(y)
                sampler = RootSamplerCategorical(y)
            else:
                sampler = root_sampler_kde(y)
            scm[node] = sampler

        else:
            X = data[:, parents]

            if is_categorical(node,INFO):
                #print(node)
                n_classes = get_num_classes(node,INFO)
                #print(n_classes)
                #print(np.unique(y))
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
                model = DiffusionRegressor(device=device,**params_regressor)
                model.fit(X, y)
                scm[node] = (model, parents, None, None)
    return scm



def _postprocess_predictions(y_preds, min_val, max_val):
        y_int = np.round(y_preds).astype(int)
        y_clipped = np.clip(y_int, min_val, max_val)
        return y_clipped


def sample_from_scm(scm, dag, n_samples, INFO) -> np.ndarray:
    n_vars = len(scm)
    data = np.zeros((n_samples, n_vars))

    for node in tqdm(nx.topological_sort(dag),desc=f'Sampling each nodes'):
        #print(f"sampling node {node}")
        model_info = scm[node]

        if callable(model_info):  # root sampler
            sampler_fn = model_info
           # print(f'root: {node}')
            # if node in Constraints_reject:
            #     constraint = Constraints_reject[node]
            #     data[:, node] = rejection_sample(sampler_fn, constraint, n_samples, max_retries)
            data[:, node] = sampler_fn.__call__(n_samples)
            if not isinstance(sampler_fn,RootSamplerCategorical):
                if INFO['col_dtype'][node] == 'int':
                    MIN = INFO['column_info'][str(node)]['min']
                    MAX = INFO['column_info'][str(node)]['max']
                    data[:,node] = _postprocess_predictions(data[:,node], min_val=int(np.floor(MIN)), max_val=int(np.ceil(MAX)))

        else:
            model, parents, noise_std, n_classes = model_info
            X_parents = data[:, parents]

            if is_categorical(node, INFO):
                dmatrix = xgb.DMatrix(X_parents)
                prob_preds = model.predict(dmatrix)
                sampled = np.array([np.random.choice(n_classes, p=probs) for probs in prob_preds])
                data[:, node] = sampled
                

            else:
                sampled = model.predict(X_parents,node=node,INFO=INFO)
                data[:, node] = sampled
    

    return data