import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from configs import Parameters
from model import MLP, Patience
from dataset import init_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import plot_metrics, calculate_metrics
import wandb


def train(config=None):
    with wandb.init(
            project="nn-project-1",
            config=config
    ) as run:
        # Set config
        config = wandb.config

        # Create dataset and dataloaders
        train, valid = init_datasets(binary=config.binary, balance=config.balance_binary)
        dl_train = DataLoader(train, batch_size=config.batch_size, shuffle=True)
        dl_valid = DataLoader(valid, batch_size=config.batch_size, shuffle=True)

        # Model setup
        device = torch.device(config.device)
        model = MLP(binary=config.binary, batch_norm=config.batch_norm).to(device)

        # Define loss and optimizer
        criterion = nn.BCELoss() if config.binary else nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

        # Early stopping
        patience = Patience(patience=config.patience)

        # Lists to store training and validation loss
        train_arr = []
        valid_arr = []

        # Variables to keep track of the best model and epoch
        best_model = model.state_dict()
        best_epoch = 0

        # Progress bar for training epochs
        p_bar = tqdm(range(1, config.epochs + 1))
        for epoch in p_bar:

            # Variables for accumulating losses
            running_train_loss = 0
            running_val_loss = 0

            # Training phase
            model.train()
            for input, ground_truth in dl_train:

                input = input.to(device)
                ground_truth = ground_truth.to(device)

                # Forward pass
                output = model(input)
                if not config.binary:
                    ground_truth = torch.argmax(ground_truth, dim=1)
                loss = criterion(output, ground_truth)
                running_train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            eval_out = []
            eval_truth = []
            with torch.no_grad():
                model.eval()
                for input, ground_truth in dl_valid:

                    input = input.to(device)
                    ground_truth = ground_truth.to(device)

                    # Forward pass
                    output = model(input)
                    if not config.binary:
                        ground_truth = torch.argmax(ground_truth, dim=1)
                    loss = criterion(output, ground_truth)
                    running_val_loss += loss.item()

                    # Save y_hat and y for metrics calculation
                    output = output.cpu()
                    ground_truth = ground_truth.cpu()

                    eval_out = np.concatenate((eval_out, np.round(output.squeeze().numpy()).astype(int)
                                              if config.binary else
                                              torch.argmax(output, dim=1)))
                    eval_truth = np.concatenate((eval_truth, np.round(ground_truth.squeeze().numpy()).astype(int)
                                                if config.binary else
                                                ground_truth))

            # Log metrics and losses to wandb
            acc, prec, recall, f1 = calculate_metrics(eval_out, eval_truth, config)
            wandb.log({"train_loss": running_train_loss / len(dl_train),
                       "valid_loss": running_val_loss / len(dl_valid),
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "Accuracy": acc,
                       "Precision": prec,
                       "Recall": recall,
                       "F1": f1,
                       }, step=epoch)

            # Update current loss statistics
            train_arr.append(running_train_loss / len(dl_train))
            valid_arr.append(running_val_loss / len(dl_valid))
            p_bar.set_postfix(train_loss=running_train_loss / len(dl_train), val_loss=running_val_loss / len(dl_valid),
                              best_loss=patience.best_loss, patience=patience.patience)

            # Check for early stopping
            do_i_stop, do_i_save = patience.should_i_stop(valid_arr[-1])
            if do_i_save:
                best_epoch = epoch
                best_model = model.state_dict()
            if do_i_stop:
                break

        # Load the best model and perform final evaluation
        model.load_state_dict(best_model)
        eval_out = []
        eval_truth = []
        with torch.no_grad():
            model.eval()
            for input, ground_truth in dl_valid:

                input = input.to(device)
                output = model(input)
                if not config.binary:
                    ground_truth = torch.argmax(ground_truth, dim=1)
                    output = torch.argmax(output, dim=1)
                else:
                    output = np.round(output.squeeze().numpy()).astype(int)
                    ground_truth = np.round(ground_truth.squeeze().numpy()).astype(int)

                # Save y_hat and y for metrics
                eval_out = np.concatenate((eval_out, output))
                eval_truth = np.concatenate((eval_truth, ground_truth))

        # Log best epoch to wandb andd plot final metrics locally (including confusion matrix)
        wandb.log({"best_epoch": best_epoch}, step=epoch)
        plot_metrics(eval_out, eval_truth, config, train_arr, valid_arr, best_epoch, epoch)

    wandb.finish()
