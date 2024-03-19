from pprint import pprint

from train import *
import wandb
import json
from configs import Parameters_sweep

if __name__ == "__main__":
    # Configuration for hyperparameter sweeping using Bayesian optimization
    sweep_config = {"method": "bayes", "name": "first_try"}
    metric = {"name": "Accuracy", "goal": "maximize"}
    sweep_config["metric"] = metric
    sweep_config["parameters"] = json.loads(Parameters_sweep().json())
    sweep_id = wandb.sweep(sweep_config, project="nn-project-1")

    # Call the agent for `train` function for each combination of hyperparameters
    wandb.agent(sweep_id, train, count=30)
