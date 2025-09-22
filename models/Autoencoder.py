import tensorflow as tf
from tensorflow.keras import layers, models

class Conv1DDenoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = models.Sequential([
            layers.Conv1D(16, kernel_size=9, padding='same', input_shape=(None, 1)),
            layers.ReLU(),
            layers.Conv1D(32, kernel_size=9, padding='same'),
            layers.ReLU()
        ])
        # Decoder
        self.decoder = models.Sequential([
            layers.Conv1D(16, kernel_size=9, padding='same'),
            layers.ReLU(),
            layers.Conv1D(1, kernel_size=9, padding='same')
        ])

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# # Example usage:
# model = Conv1DDenoiser()
# # For a batch of signals with shape (batch, length, channels)
# dummy_input = tf.random.normal([1, 1024, 1])  
# output = model(dummy_input)
# print(output.shape)
