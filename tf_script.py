import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data
def generate_data(num_samples=1000, noise_factor=0.5):
    np.random.seed(42)
    t = np.linspace(0, 1, num_samples)
    s = np.sin(2 * np.pi * t)  # Clean signal
    n = noise_factor * np.random.normal(size=num_samples)  # Noise
    x = s + n  # Noisy signal
    return x, s

# Generate data
x_train, s_train = generate_data(num_samples=1000)
x_test, s_test = generate_data(num_samples=300)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, s_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(x_test, s_test)
print(f"Test Loss: {loss}")

# Predict and plot results
import matplotlib.pyplot as plt

s_pred = model.predict(x_test)

model.save('adaptive_filter.h5')

# plt.figure(figsize=(12, 6))
# plt.plot(x_test, label='Noisy Signal')
# plt.plot(s_test, label='Clean Signal')
# plt.plot(s_pred, label='Predicted Clean Signal', linestyle='dashed')
# plt.legend()
# plt.show()