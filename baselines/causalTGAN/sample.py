import os
import pickle
import torch
import numpy as np
import argparse
import pandas as pd


from baselines.causalTGAN.model.causalTGAN import load_model
from baselines.causalTGAN.helper.utils import restore_feature_info


def main(args):

    model_path = f"baselines/causalTGAN/ckpts/{args.dataname}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    transformer, feature_info, causal_graph = restore_feature_info(model_path)
    model, _ = load_model(model_path, device, feature_info, transformer)

    if model.causal_controller is not None:
        model.causal_controller.set_causal_mechanisms_eval()
    if model.condGAN is not None:
        model.condGAN.generator.eval()

    r = []
    train_df = pd.read_csv(f"data/{args.dataname}/train.csv")

    gen_num = train_df.shape[0]
    batch_size = 1000
    

    num_full_batches = gen_num // batch_size # Integer division: 1500 // 1000 = 1
    remaining_samples = gen_num % batch_size # Remainder: 1500 % 1000 = 500

    with torch.no_grad():
        for _ in range(num_full_batches):
            samples = model.sample(batch_size)
            r.append(samples.cpu())

        if remaining_samples > 0:
            samples = model.sample(remaining_samples)
            r.append(samples.cpu())

    samples = torch.cat(r, dim=0)
    print(samples.shape)
    print(transformer)
    sample_df = transformer.inverse_transform(samples.data.numpy())
    

    sample_df.to_csv(args.save_path, index=False)

if __name__ == '__main__':
    main()

