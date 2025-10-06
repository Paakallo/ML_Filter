from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt("STM_data.txt")


plt.plot(data)
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title("Reconstructed Signal")
plt.legend()
plt.grid(True)
plt.show()

