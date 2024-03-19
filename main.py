from train import *
from configs import Parameters

if __name__ == "__main__":

    # Get parameters for dataset, model and training and run the train with wandb

    config = Parameters()
    train(config=config)
