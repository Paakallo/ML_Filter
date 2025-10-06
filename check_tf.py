import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="conv_downsize.tflite", experimental_delegates=[])
interpreter.allocate_tensors()
for tensor in interpreter.get_tensor_details():
    print(tensor['name'])

print('Printing ops \n')

for op_detail in interpreter._get_ops_details():
    print(op_detail['op_name'])