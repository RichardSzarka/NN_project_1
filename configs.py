from pydantic import BaseModel
import torch
from torch.optim import SGD


class Parameters(BaseModel):
    # Dataset parameters:
    binary: bool = False
    balance_binary: bool = False

    # Training parameters:
    lr: float = 3e-4
    epochs: int = 10000
    batch_size: int = 16
    patience: int = 150
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model parameters:
    batch_norm: bool = True
    model_config: dict = {"arbitrary_types_allowed": True}


class Parameters_sweep(BaseModel):
    # Dataset parameters for hyperparameter sweeping:
    binary: dict = {"values": [0]}
    balance_binary: dict = {"values": [0]}

    # Training parameters for hyperparameter sweeping:
    lr: dict = {"values": [0.03, 3e-5, 3e-2]}
    epochs: dict = {"values": [200]}
    batch_size: dict = {"values": [16, 32, 64]}
    patience: dict = {"values": [150]}
    device: dict = {"values": ["cuda" if torch.cuda.is_available() else "cpu"]}

    # Model parameters for hyperparameter sweeping:
    batch_norm: dict = {"values": [1, 0]}
    model_config: dict = {"arbitrary_types_allowed": True}
