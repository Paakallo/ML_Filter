from matplotlib import pyplot as plt
import numpy as np

input_data = np.loadtxt("input_data.txt")
ideal_data = np.loadtxt("ideal_data.txt")
rec_data = np.loadtxt("rec_data.txt")

plt.figure(figsize=(12,6))
plt.plot(input_data, label="Noise signal")
plt.plot(ideal_data, label="Ideal signal")
plt.plot(rec_data, label="Denoised signal")
plt.xlabel("Sample index")
plt.title("Denoised sine wave")
plt.legend()
plt.grid(True)
plt.show()

