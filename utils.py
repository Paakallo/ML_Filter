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

