"""
Training script
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import torch
import datetime as dt
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from FLightcase.utils.deep_learning.model import copy_net
from FLightcase.utils.deep_learning.evaluation import evaluate


def get_criterion(criterion_txt):
    if criterion_txt == 'l1loss':
        return nn.L1Loss(reduction='sum')
    else:
        raise ValueError(f'Cannot find criterion for {criterion_txt}')


def get_optimizer(optimizer_txt, net, lr):
    if optimizer_txt == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError(f'Cannot find optimizer for {optimizer_txt}')


def print_epoch_message(epoch, lr, train_loss, train_mae, val_loss, val_mae, best_loss, n_worse_epochs):

    epoch_message = f'Epoch: {epoch} || ' \
                    f'lr: {lr:.2E} || ' \
                    f'train loss: {train_loss:.2f} || ' \
                    f'train mae: {train_mae:.2f} || ' \
                    f'val loss: {val_loss:.2f} || ' \
                    f'val mae: {val_mae:.2f} || ' \
                    f'best loss {best_loss:.2f} || ' \
                    f'number of epochs without improvement: {n_worse_epochs}'
    print(epoch_message)


def train(n_epochs, device, train_loader, val_loader, optimizer, net, criterion, scheduler, save_best_sd=False,
          state_dict_output_dir_path=None, patience_stop=None):
    """ Training function

    :param n_epochs: int, number of epochs
    :param device: torch.device, cpu or gpu
    :param train_loader: torch DataLoader, training data
    :param val_loader: torch DataLoader, validation data
    :param optimizer: torch optimizer
    :param net: torch neural network
    :param criterion: torch.nn loss criterion. CAVE: use "sum" as reduction!
    :param scheduler: torch lr scheduler
    :param save_best_sd: bool, save the best state dict?
    :param state_dict_output_dir_path: str, path to directory to write state dicts to
    :param patience_stop: int, number of epochs without improvement after which training will be interrupted
    :return: str, path to state dict of the best model
    """

    # Verify whether output directory specified when saving state dicts
    if save_best_sd and state_dict_output_dir_path is None:
        raise ValueError('Please specify state_dict_output_dir_path')

    # Initialize variables
    best_loss = 1e9
    n_worse_epochs = 0
    best_model = None
    best_model_path = None
    train_loss_list = []
    val_loss_list = []

    for epoch in range(n_epochs):
        # Initialize variables
        train_loss_sum = 0
        samples_count = 0
        train_true_label_list = []
        train_pred_label_list = []

        # Set model to training mode and send to GPU
        net.train().to(device)

        for neuro_data_list, label in tqdm(train_loader):
            # Send label to device. Send other data to device when passing to net
            label = label.to(device)

            # Clear gradient
            optimizer.zero_grad()

            # Predict label, calculate loss and calculate gradient
            pred_label = net(*[i.to(device) for i in neuro_data_list])      # Unwrap list when passing to function
            pred_label = torch.squeeze(pred_label, dim=1)                   # Remove one dim to match label dim
            loss = criterion(pred_label, label)
            loss.backward()

            # Extend lists with true and predicted label
            train_pred_label_list.extend(pred_label.tolist())
            train_true_label_list.extend(label.tolist())

            # Sum variables to allow calculation of mean loss per epoch
            samples_count += neuro_data_list[0].shape[0]   # First dimension is number of samples in batch
            train_loss_sum += loss.detach().item()

            # Update weights
            optimizer.step()

        # Calculate train metrics:
        # - Mean loss per epoch
        # - MAE
        train_loss = train_loss_sum / samples_count
        train_mae = mean_absolute_error(train_true_label_list, train_pred_label_list)

        # Validation
        val_loss, val_true_label_list, val_pred_label_list = evaluate(net, val_loader, criterion, device, 'validation')
        val_mae = mean_absolute_error(val_true_label_list, val_pred_label_list)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Evaluate whether loss is lower.
        # - If lower, save state dict and reset number of bad epochs
        # - If equal or higher, increase bad epochs with one
        if val_loss < best_loss:
            best_loss = val_loss
            n_worse_epochs = 0
            best_model = copy_net(net)
            if save_best_sd:
                best_model_path = os.path.join(
                    state_dict_output_dir_path,
                    f'best_model_epoch={epoch}_time={str(dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))}.pt'
                )
        else:
            n_worse_epochs += 1

        # Get learning rate (might have changed during training)
        lr = optimizer.param_groups[0]['lr']

        # Add loss to list
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Epoch print message
        print_epoch_message(epoch, lr, train_loss, train_mae, val_loss, val_mae, best_loss, n_worse_epochs)

        # Stop after predefined number of bad epochs
        if patience_stop is not None:
            if n_worse_epochs == patience_stop:
                print(f'Stopping early after {epoch+1} epochs')  # Epochs start counting from 0
                break

    # Save best state dict if desired
    if save_best_sd:
        torch.save(best_model.state_dict(), best_model_path)

    return best_model, best_loss, train_loss_list, val_loss_list
