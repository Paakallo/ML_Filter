import torch
import torch.nn as nn

class Conv1DDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
