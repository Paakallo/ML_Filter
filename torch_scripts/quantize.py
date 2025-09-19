from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "results/autoencoder.onnx"
output_model = "no_preproc_autoencoder_int8.onnx"


quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # or QuantType.QUInt8
)
