import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()
