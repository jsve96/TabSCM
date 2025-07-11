import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
from tqdm import tqdm


# class MLPConditionedUNet(nn.Module):
#     def __init__(self, input_dim, cond_dim, hidden_dim=256):
#         super().__init__()
#         # 1 = scalar y, cond_dim = parents (X features), 1 = timestep
#         self.net = nn.Sequential(
#             nn.Linear(input_dim + cond_dim + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             #nn.ReLU(),
#             #nn.Linear(hidden_dim,32),
#             #nn.ReLU(),
#             #nn.Linear(32,hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)  # predict noise for input_dim (which is 1 for scalar y)
#         )

#     def forward(self, y, cond, t):
#         # y: (B, input_dim), cond: (B, cond_dim), t: (B, 1)
#         return self.net(torch.cat([y, cond, t], dim=-1))


class FourierTimestepEmbed(nn.Module):
    def __init__(self, embed_dim, scale=10):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        # t: (B, 1)
        angles = 2 * np.pi * t @ self.freqs.unsqueeze(0)  # (B, embed_dim/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)




class MLPConditionedUNet(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_embed = FourierTimestepEmbed(embed_dim=time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, y, cond, t):
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        x = torch.cat([y, cond, t_emb], dim=-1)
        return self.net(x)




class DiffusionRegressor:
    def __init__(self, timesteps, epochs, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.model = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.alpha_hat = None
        self.beta = None
        self.epochs = epochs

    def save(self, experiment, node):
        os.makedirs(f'{experiment}', exist_ok=True)

        # Save model weights (if fitted)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(f'{experiment}', "model.pt"))

        # Save config/state separately
        state = {
            "timesteps": self.timesteps,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "X_scaler": self.X_scaler,
            "y_scaler": self.y_scaler,
            "beta": self.beta.cpu().numpy() if self.beta is not None else None,
            "alpha_hat": self.alpha_hat.cpu().numpy() if self.alpha_hat is not None else None,
        }

        with open(os.path.join(f'{experiment}', "state.pkl"), "wb") as f:
            pickle.dump(state, f)

    def _postprocess_predictions(self,y_preds, min_val, max_val):
        y_int = np.round(y_preds).astype(int)
        y_clipped = np.clip(y_int, min_val, max_val)
        return y_clipped


    def _prepare_noise_schedule(self):
        beta = np.linspace(1e-4, 0.02, self.timesteps)
        alpha = 1 - beta
        alpha_hat = np.cumprod(alpha)
        return beta, alpha, alpha_hat
    
   

    def fit(self, X, y, lr=1e-3, batch_size=512):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        X = np.array(X).astype('float32')
        y = np.array(y).astype('float32')
        X_scaled = self.X_scaler.fit_transform(X)
        self.y_min = y.min()
        self.y_max = y.max()

        
        eps_scale = 1e-6
        y_scaled = (y - self.y_min) / (self.y_max - self.y_min)
        y_scaled = np.clip(y_scaled, eps_scale, 1 - eps_scale)
        y_scaled = np.log(y_scaled / (1 - y_scaled))

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(-1).to(self.device) 
        n_samples, cond_dim = X_tensor.shape
        input_dim = y_tensor.shape[-1] 

        self.model = MLPConditionedUNet(input_dim=input_dim, cond_dim=cond_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        
        total_steps = self.epochs * (n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


        beta, alpha, alpha_hat = self._prepare_noise_schedule()
        self.beta = torch.tensor(beta, dtype=torch.float32).to(self.device)
        self.alpha_hat = torch.tensor(alpha_hat, dtype=torch.float32).to(self.device)

   
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(self.epochs), desc="Fit Regressor", position=1, leave=False):
            for batch_X, batch_y in dataloader:
                current_batch_size = batch_X.shape[0]

                t = torch.randint(0, self.timesteps, (current_batch_size,), device=self.device)
                noise = torch.randn_like(batch_y)
                
                sqrt_alpha_hat = self.alpha_hat[t].unsqueeze(-1).sqrt()
                sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]).unsqueeze(-1).sqrt()

                y_noisy = sqrt_alpha_hat * batch_y + sqrt_one_minus_alpha_hat * noise
                
                t_norm = t.float() / self.timesteps
                t_norm = t_norm.unsqueeze(-1) # (B, 1)

                noise_pred = self.model(y_noisy, batch_X, t_norm)
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                scheduler.step()

    def sample(self, X, n_steps=None):
        """
        X: (n_samples, n_features) - unscaled input
        Returns y_sampled: (n_samples,) - denormalized output
        """
        if self.model is None:
            raise ValueError("Model not fitted.")

        X = np.array(X).astype('float32')
        X_scaled = self.X_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        n_samples = X_tensor.shape[0]

        n_steps = self.timesteps if n_steps is None else n_steps

        
        y = torch.randn(n_samples, 1).to(self.device)
        self.model.eval()
        with torch.no_grad(): 
            for t_idx in reversed(range(n_steps)):
                t_tensor = torch.tensor([t_idx / self.timesteps], dtype=torch.float32).repeat(n_samples, 1).to(self.device)
                
                beta_t = self.beta[t_idx]
                alpha_hat_t = self.alpha_hat[t_idx]
                
                noise_pred = self.model(y, X_tensor, t_tensor)
                
                eps = 1e-6
                denom_b = torch.sqrt(1 - beta_t + eps)
                denom_a = torch.sqrt(1 - alpha_hat_t + eps)
                y = (1 / denom_b) * (y - beta_t / denom_a * noise_pred)

                
                if t_idx > 0: # Add noise for intermediate steps
                    y += torch.randn_like(y) * beta_t.sqrt()
        
        # Denormalize the sampled y before returning
        y_np = y.cpu().detach().numpy()
        y_sigmoid = 1 / (1 + np.exp(-y_np))  # sigmoid
        y_denormalized = self.y_min + y_sigmoid * (self.y_max - self.y_min)


        return y_denormalized.squeeze() # Squeeze back to (n_samples,)

    def predict(self, X,node,INFO, n_samples_per_x=1):
        """
        Predict the mean of multiple samples for a more stable point estimate.
        X: (n_samples, n_features)
        n_samples_per_x: Number of samples to draw for each X to average.
        Returns y_mean_prediction: (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        X = np.array(X).astype('float32')
        X_repeated = np.repeat(X, n_samples_per_x, axis=0)
        
        all_samples = self.sample(X_repeated)
        
        n_original_samples = X.shape[0]
        y_predictions = all_samples.reshape(n_original_samples, n_samples_per_x)

        y_mean = y_predictions.mean(axis=1)
        
        if INFO['col_dtype'][node] == 'int':
            MIN = INFO['column_info'][str(node)]['min']
            MAX = INFO['column_info'][str(node)]['max']
            return self._postprocess_predictions(y_mean, min_val=int(np.floor(MIN)), max_val=int(np.ceil(MAX)))
        
        return y_mean