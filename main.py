import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import os

class DSAN(nn.Module):
    def __init__(self):
        super(DSAN, self).__init__()
        self.embedding = nn.Linear(3, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(18 * 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        return self.mlp(x)

_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DSAN().to(_DEVICE)
        if os.path.exists('model.pt'):
            _MODEL.load_state_dict(torch.load('model.pt', map_location=_DEVICE, weights_only=True))
        _MODEL.eval()
    return _MODEL

def your_algorithm(d_hat_u, p_bs):
    model = load_model()
    
    bs_x_min, bs_x_max = p_bs[0, :].min(), p_bs[0, :].max()
    bs_y_min, bs_y_max = p_bs[1, :].min(), p_bs[1, :].max()
    
    diag_dist = np.sqrt((bs_x_max - bs_x_min)**2 + (bs_y_max - bs_y_min)**2)
    max_valid_dist = diag_dist * 1.2
    
    margin_x = (bs_x_max - bs_x_min) * 0.1
    margin_y = (bs_y_max - bs_y_min) * 0.1
    scale_x_min, scale_x_max = bs_x_min - margin_x, bs_x_max + margin_x
    scale_y_min, scale_y_max = bs_y_min - margin_y, bs_y_max + margin_y
    
    d_hat_filtered = d_hat_u.copy()
    invalid_mask = np.isnan(d_hat_filtered) | (d_hat_filtered <= 0) | (d_hat_filtered > max_valid_dist)
    d_hat_filtered[invalid_mask] = max_valid_dist
    d_hat_scaled = d_hat_filtered / max_valid_dist
    
    bs_scaled = np.zeros_like(p_bs)
    bs_scaled[0, :] = (p_bs[0, :] - scale_x_min) / (scale_x_max - scale_x_min)
    bs_scaled[1, :] = (p_bs[1, :] - scale_y_min) / (scale_y_max - scale_y_min)
    
    features = np.zeros((18, 3))
    features[:, 0] = bs_scaled[0, :]
    features[:, 1] = bs_scaled[1, :]
    features[:, 2] = d_hat_scaled
    
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).squeeze(0).cpu().numpy()
        
    pred_x = pred_scaled[0] * (scale_x_max - scale_x_min) + scale_x_min
    pred_y = pred_scaled[1] * (scale_y_max - scale_y_min) + scale_y_min
    
    return np.array([pred_x, pred_y])

def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)
        
    return p_hat

if __name__ == "__main__":
    main()
