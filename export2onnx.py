import torch
import numpy as np
from models.Autoencoder import Conv1DDenoiser

model_name = "autoencoder"
window_size = 1024

# torch_model = Conv1DDenoiser()
torch_model = torch.load(f"results/{model_name}.pth")

# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
dummy_input = torch.randn(1, 1, window_size, dtype=torch.float32)
onnx_program = torch.onnx.export(torch_model, 
                                 dummy_input,
                                 f=f"{model_name}.onnx",
                                 export_params=True,
                                 input_names=['input'],
                                 output_names=['output']
                                 )

# onnx_program.save(f"{model_name}.onnx")