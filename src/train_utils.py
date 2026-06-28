# Inicializa as bibliotecas necessárias
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DTYPE = torch.float32

# -----------------------------
# Dataset
# -----------------------------
class DataProcess(Dataset):
    def __init__(self, df, feat_cols, target_cols=None, x_scaler=None, y_scaler=None):
        self.X = df[feat_cols].to_numpy(dtype=np.float32, copy=True)
        self.has_y = target_cols is not None
        self.Y = (
            df[target_cols].to_numpy(dtype=np.float32, copy=True)
            if self.has_y
            else None
        )

        # scaler só para features (z-score)
        if x_scaler is None:
            mu = self.X.mean(axis=0, keepdims=True)
            sd = self.X.std(axis=0, keepdims=True) + 1e-8
            self.x_mu = mu
            self.x_sd = sd
        else:
            self.x_mu, self.x_sd = x_scaler

        if self.has_y:
            if y_scaler is None:
                y_mu = self.Y.mean(axis=0, keepdims=True)
                y_sd = self.Y.std(axis=0, keepdims=True) + 1e-8
                self.y_mu = y_mu
                self.y_sd = y_sd
            else:
                self.y_mu, self.y_sd = y_scaler

            self.Yn = (self.Y - self.y_mu) / self.y_sd
        else:
            self.y_mu = None
            self.y_sd = None
            self.Yn = None

        # normaliza X e Y uma única vez para reduzir overhead por batch
        self.Xn = torch.from_numpy((self.X - self.x_mu) / self.x_sd).to(DTYPE)
        if self.has_y:
            self.Yn = torch.from_numpy((self.Y - self.y_mu) / self.y_sd).to(DTYPE)

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        if self.has_y:
            return self.Xn[idx], self.Yn[idx]
        return self.Xn[idx]


# -----------------------------
# 2) Arquitetura Geral MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=6, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # init leve (ajuda tanh)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, model_path, device, patience_es=20):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_path = model_path
        self.device = device

        self.best_val = float("inf")
        self.bad = 0
        self.patience_es = patience_es
    

    # Treino em batch supervisionado
    def train_supervised_epoch(self, dl_data):
        self.model.train()
        
        total_loss = 0.0
        n_train = 0

        for batch in dl_data:
            Xn, Yn = batch
            Xn = Xn.to(self.device, non_blocking=True)
            Yn = Yn.to(self.device, non_blocking=True)

            # =========================
            # (Loss de dados (ΔU)
            # =========================
            pred_data = self.model(Xn)  # forward normal
            loss = torch.mean((pred_data - Yn) ** 2)   # pred_data é (N,3) e Yn é (N,3)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * Xn.size(0)
            n_train += Xn.size(0)
        return total_loss / n_train

    # Avaliação da loss com dados de teste
    def validate_supervised(self, dl_val):
        self.model.eval()
        val_total = 0.0
        n_val = 0
        with torch.no_grad():
            for Xv, Yv in dl_val:
                Xv = Xv.to(self.device, non_blocking=True)
                Yv = Yv.to(self.device, non_blocking=True)

                pred_v = self.model(Xv)
                loss_v = torch.mean((pred_v - Yv) ** 2)

                val_total += loss_v.item() * Xv.size(0)
                n_val += Xv.size(0)

        return val_total / n_val

    # Update do modelo com melhor loss
    def update_best_model(self, loss_val):
        improved = loss_val < self.best_val

        if improved:
                self.best_val = loss_val
                self.bad = 0
                torch.save(self.model.state_dict(), self.model_path )
        else:
            self.bad += 1

        return improved
    
    def stop_improve(self):
        return self.bad >= self.patience_es
    
    def load_best(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return self.model
