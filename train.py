import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from models.Autoencoder import Conv1DDenoiser
from utils import generate_data

model = Conv1DDenoiser()
dataset = 'autoencoder'
graph_name = 'autoencoder'
model_name = "autoencoder"

if not os.path.exists(f'dataset/{dataset}'):
    os.makedirs(f'dataset/{dataset}', exist_ok=True)

x_train, s_train = generate_data(100000)
x_test, s_test = generate_data(300)

window = 1024  # example window length
stride = 512
segments = []
targets = []
for start in range(0, len(x_train)-window, stride):
    segments.append(x_train[start:start+window])
    targets.append(s_train[start:start+window])

x_train_t = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)  # [N, 1, window]
s_train_t = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

t_segments = []
t_segments.append(x_test)
t_targets = []
t_targets.append(s_test)

# format to models standard
x_test_t = torch.tensor(t_segments, dtype=torch.float32).unsqueeze(1)
s_test_t = torch.tensor(t_targets, dtype=torch.float32).unsqueeze(1) 

train_ds = TensorDataset(x_train_t, s_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# --- Loss & optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

torch.save(model, f"results/{model_name}.pth")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    s_pred_t = model(x_test_t)
    test_loss = criterion(s_pred_t, s_test_t).item()

print(f"Test Loss: {test_loss:.4f}")

s_pred = s_pred_t.squeeze().numpy()

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(x_test, label='Noisy Signal')
plt.plot(s_test, label='Clean Signal')
plt.plot(s_pred, label='Predicted Clean Signal', linestyle='dashed')
plt.legend()
plt.savefig(f'results/{graph_name}.png')
