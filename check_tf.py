import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="autoencoder_filter.tflite")
interpreter.allocate_tensors()
for tensor in interpreter.get_tensor_details():
    print(tensor['name'])
