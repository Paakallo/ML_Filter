import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="best_sequence_model.tflite", experimental_delegates=[])
interpreter.allocate_tensors()
for tensor in interpreter.get_tensor_details():
    print(tensor['name'])

print('Printing ops \n')

for op_detail in interpreter._get_ops_details():
    print(op_detail['op_name'])

for input in interpreter.get_input_details():
    print(input['index'])