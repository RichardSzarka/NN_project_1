from pydantic import BaseModel
import torch
from torch.optim import SGD


class Parameters(BaseModel):
    lr: float = 6e-3
    epochs: int = 10000
    batch_size: int = 32
    binary: bool = True
    balance_binary: bool = True
    batch_norm: bool = True
    patience: int = 150
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_config: dict = {"arbitrary_types_allowed": True}


class Parameters_sweep(BaseModel):
    lr:dict = {"min": 0.00001, "max": 0.01}
    epochs: dict = {"values": [10000]}
    batch_size: dict = {"values": [8, 16, 32, 64, 128]}
    binary: dict = {"values": [0]}
    balance_binary: dict = {"values": [0]}
    batch_norm: dict = {"values": [1, 0]}
    patience: dict = {"values": [150]}
    device: dict = {"values": ["cuda" if torch.cuda.is_available() else "cpu"]}
    model_config: dict = {"arbitrary_types_allowed": True}
