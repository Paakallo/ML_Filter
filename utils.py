import numpy as np


def generate_data(num_samples=1000, noise_factor=0.5):
    np.random.seed(42)
    t = np.linspace(0, 1, num_samples, dtype=np.float32)
    s = np.sin(2 * np.pi * t,dtype=np.float32)  # Clean signal
    n = noise_factor * np.random.normal(size=num_samples)  # Noise
    x = s + n  # Noisy signal
    x.astype(dtype=np.float32)
    s.astype(dtype=np.float32)
    return x, s


def create_windows(x, s, window_size=10):
    X, Y = [], []
    for i in range(window_size, len(x)):
        X.append(x[i-window_size:i])  # 10 samples (previous 9 + current)
        Y.append(s[i])                # clean target for the current sample
    X = np.array(X)[..., np.newaxis]  # shape: (samples, 10, 1)
    Y = np.array(Y)[..., np.newaxis]  # shape: (samples, 1)
    return X.astype(np.float32), Y.astype(np.float32)
