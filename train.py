import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AttentionPositioningDataset(Dataset):
    def __init__(self, mat_path):
        data = sio.loadmat(mat_path, squeeze_me=False)
        self.BS_positions = np.asarray(data['BS_positions'], dtype=float)
        self.d_hat = np.asarray(data['d_hat'], dtype=float)
        self.p = np.asarray(data['p'], dtype=float)
        self.num_user = self.d_hat.shape[1]

        bs_x_min, bs_x_max = self.BS_positions[0, :].min(), self.BS_positions[0, :].max()
        bs_y_min, bs_y_max = self.BS_positions[1, :].min(), self.BS_positions[1, :].max()
        
        diag_dist = np.sqrt((bs_x_max - bs_x_min)**2 + (bs_y_max - bs_y_min)**2)
        self.max_valid_dist = diag_dist * 1.2 
        
        invalid_mask = np.isnan(self.d_hat) | (self.d_hat <= 0) | (self.d_hat > self.max_valid_dist)
        self.d_hat[invalid_mask] = self.max_valid_dist

        self.d_hat_scaled = self.d_hat / self.max_valid_dist
        
        margin_x = (bs_x_max - bs_x_min) * 0.1
        margin_y = (bs_y_max - bs_y_min) * 0.1
        self.x_min, self.x_max = bs_x_min - margin_x, bs_x_max + margin_x
        self.y_min, self.y_max = bs_y_min - margin_y, bs_y_max + margin_y
        
        self.BS_scaled = np.zeros_like(self.BS_positions)
        self.BS_scaled[0, :] = (self.BS_positions[0, :] - self.x_min) / (self.x_max - self.x_min)
        self.BS_scaled[1, :] = (self.BS_positions[1, :] - self.y_min) / (self.y_max - self.y_min)

        self.p_scaled = np.zeros_like(self.p)
        self.p_scaled[0, :] = (self.p[0, :] - self.x_min) / (self.x_max - self.x_min)
        self.p_scaled[1, :] = (self.p[1, :] - self.y_min) / (self.y_max - self.y_min)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        features = np.zeros((18, 3))
        features[:, 0] = self.BS_scaled[0, :]
        features[:, 1] = self.BS_scaled[1, :]
        features[:, 2] = self.d_hat_scaled[:, idx]
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(self.p_scaled[:, idx], dtype=torch.float32)
        return features_tensor, labels_tensor

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
        out = self.mlp(x)
        return out

def main():
    mat_path = 'DH_FR1.mat'
    try:
        dataset = AttentionPositioningDataset(mat_path)
    except FileNotFoundError:
        print(f"[오류] '{mat_path}' 파일을 찾을 수 없습니다.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 사용 중인 디바이스: {device}")
    
    model = DSAN().to(device)
    
    criterion = nn.HuberLoss(delta=1.0) 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    num_epochs = 200
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), 'model.pt')
    print("\n[*] 학습 완료! 'model.pt' 저장됨.")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()
