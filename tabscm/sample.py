from model import *
#### need to be changed to def main(args)....

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
            data[:, node] = sampler_fn(n_samples)
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