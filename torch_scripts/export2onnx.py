import torch
import numpy as np
from models.Autoencoder_torch import Conv1DDenoiser

model_name = "adaptive_filter"

# torch_model = Conv1DDenoiser()
torch_model = torch.load(f"results/{model_name}.pth", weights_only=False)

if model_name == "autoencoder":
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
    window_size = 1024
    dummy_input = torch.randn(1, 1, window_size, dtype=torch.float32) #TODO: check if this dummy input is correct
else:
    probe_num = 10000
    dummy_input = torch.randn(1, 64, 1, dtype=torch.float32) #TODO: this maybe wrong

onnx_program = torch.onnx.export(torch_model, 
                                 dummy_input,
                                 f=f"{model_name}.onnx",
                                 export_params=True,
                                 input_names=['input'],
                                 output_names=['output']
                                 )
