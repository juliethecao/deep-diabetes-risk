import torch

DATA_DIR = "../data_processed/"

def load_split(split):
    data = torch.load(f"{DATA_DIR}/{split}.pt")
    return data["X"], data["y"]