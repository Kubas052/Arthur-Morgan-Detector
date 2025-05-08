import torch
import torch_directml

device = torch_directml.device()
print(f"DirectML : {torch_directml.is_available()}")
print(f"GPU: {[torch_directml.device(i) for i in range(torch_directml.device_count())]}")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)
_ = torch.matmul(a, b)
print("GPU performance test completed.")