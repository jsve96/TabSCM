import networkx as nx
import numpy as np





def generate_dag_from_dict(data):
    G = nx.DiGraph()

    for node in data['nodes']:
        G.add_node(node)
    for i,j in data['edges']:
        G.add_edge(i,j)

    return G



import torch
import xgboost as xgb
from tabscm.model import *
from tabscm.diffusion_regressor import *
import json
import pickle
import os

def load_scm(load_dir: str, experiment: str = "default", device: str = "cpu"):
    from tabscm.diffusion_regressor import DiffusionRegressor  # Replace with actual module
    
    with open(os.path.join(load_dir, f"{experiment}_scm_meta.json"), "r") as f:
        scm_meta = json.load(f)

    scm = {}
    for node, meta in scm_meta.items():
        #print(node)
        model_type = meta["model_type"]
        parents = meta["parents"]
        n_classes = meta["n_classes"]
        model_path = os.path.join(load_dir, meta["model_path"])

        if model_type == "sampler":
            with open(model_path, "rb") as f:
                sampler = pickle.load(f)
            scm[int(node)] = sampler

        elif model_type == "xgb":
            model = xgb.Booster()
            model.load_model(os.path.join(model_path, "xgb_model.json"))
            scm[int(node)] = (model, parents, None, n_classes)

        elif model_type == "diffusion":
            with open(os.path.join(model_path, "state.pkl"), "rb") as f:
                state = pickle.load(f)

            model = DiffusionRegressor(
                timesteps=state["timesteps"],
                epochs=0,  # Not training anymore
                device=device
            )
            model.X_scaler = state["X_scaler"]
            model.y_scaler = state["y_scaler"]
            model.y_min = state["y_min"]
            model.y_max = state["y_max"]
            model.beta = torch.tensor(state["beta"], dtype=torch.float32).to(device)
            model.alpha_hat = torch.tensor(state["alpha_hat"], dtype=torch.float32).to(device)

            # Rebuild network
            input_dim = 1
            cond_dim = len(parents)
            model.model = MLPConditionedUNet(input_dim=input_dim, cond_dim=cond_dim).to(device)
            model.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device))
            model.model.eval()

            scm[int(node)] = (model, parents, None, n_classes)

        else:
            raise ValueError(f"Unknown model type: {model_type}")
        #print(f'{node} finished')

    return scm




