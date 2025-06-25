import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")