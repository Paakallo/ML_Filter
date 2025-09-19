import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --- Data generation ---
def generate_data(num_samples=1000, noise_factor=0.5):
    np.random.seed(42)
    t = np.linspace(0, 1, num_samples)
    s = np.sin(2 * np.pi * t)  # Clean signal
    n = noise_factor * np.random.normal(size=num_samples)  # Noise
    x = s + n  # Noisy signal
    return x, s

x_train, s_train = generate_data(100000)
x_test, s_test = generate_data(300)

# Convert to torch tensors, shape [N,1]
x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
s_train_t = torch.tensor(s_train, dtype=torch.float32).unsqueeze(1)
x_test_t  = torch.tensor(x_test,  dtype=torch.float32).unsqueeze(1)
s_test_t  = torch.tensor(s_test,  dtype=torch.float32).unsqueeze(1)

# save dataset
torch.save(x_train_t, "dataset/x_train.pt")
torch.save(s_train_t, "dataset/s_train.pt")
torch.save(x_test_t, "dataset/x_test.pt")
torch.save(s_test_t, "dataset/s_test.pt")

train_ds = TensorDataset(x_train_t, s_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# --- Model ---
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = DenoiseNet()

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

torch.save(model, "denoise.pth")

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
plt.savefig('graph.png')
