from utils import *
from models.AdaptiveFilter import DenoiseNet
import torch

x_train, s_train = generate_data(100000)
x_test, s_test = generate_data(300)

# Convert to torch tensors, shape [N,1]
x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
s_train_t = torch.tensor(s_train, dtype=torch.float32).unsqueeze(1)
x_test_t  = torch.tensor(x_test,  dtype=torch.float32).unsqueeze(1)
s_test_t  = torch.tensor(s_test,  dtype=torch.float32).unsqueeze(1)

print("Input tensor: ", x_test_t.shape)

model = DenoiseNet()

print("Model layers: ")

for param in model.parameters():
    print(type(param), param.size())

dummy_input = torch.randn(1, 1, 10000, dtype=torch.float32)
print(dummy_input.size())

