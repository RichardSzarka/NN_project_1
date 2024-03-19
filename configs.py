from pydantic import BaseModel
import torch


class Parameters(BaseModel):
    # Dataset parameters:
    binary: bool = True
    balance_binary: bool = False

    # Training parameters:
    lr: float = 6e-3
    epochs: int = 10000
    batch_size: int = 32
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
    lr:dict = {"min": 0.00001, "max": 0.01}
    epochs: dict = {"values": [10000]}
    batch_size: dict = {"values": [8, 16, 32, 64, 128]}
    patience: dict = {"values": [150]}
    device: dict = {"values": ["cuda" if torch.cuda.is_available() else "cpu"]}

    # Model parameters for hyperparameter sweeping:
    batch_norm: dict = {"values": [1, 0]}
    model_config: dict = {"arbitrary_types_allowed": True}
