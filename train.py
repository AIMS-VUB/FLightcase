"""
Training script
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import torch
import argparse
import pandas as pd
import datetime as dt
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from monai.networks.nets import DenseNet
from DL_utils.data import get_data_loader, split_data
from DL_utils.model import get_weights
from DL_utils.evaluation import evaluate


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


def train(n_epochs, device, train_loader, val_loader, optimizer, net, criterion, scheduler, state_dict_output_dir_path):
    """ Training function

    :param n_epochs: int, number of epochs
    :param device: torch.device, cpu or gpu
    :param train_loader: torch DataLoader, training data
    :param val_loader: torch DataLoader, validation data
    :param optimizer: torch optimizer
    :param net: torch neural network
    :param criterion: torch.nn loss criterion
    :param scheduler: torch lr scheduler
    :param state_dict_output_dir_path: str, path to directory to write state dicts to
    :return: str, path to state dict of the best model
    """
    # Initialize variables
    best_loss = 1e9
    n_worse_epochs = 0
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

        for img, label in tqdm(train_loader):
            # Send to device
            img = img.to(device)
            label = label.to(device)

            # Clear gradient
            optimizer.zero_grad()

            # Predict label, calculate loss and calculate gradient
            pred_label = net(img)
            pred_label = torch.squeeze(pred_label, dim=1)  # Remove one dimension to match dimensionality of label
            loss = criterion(pred_label, label)
            loss.backward()

            # Extend lists with true and predicted label
            train_pred_label_list.extend(pred_label.tolist())
            train_true_label_list.extend(label.tolist())

            # Sum variables to allow calculation of mean loss per epoch
            samples_count += img.shape[0]  # First dimension is number of samples in batch
            train_loss_sum += loss.sum().detach().item()

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
        scheduler.step(val_loss)

        # Evaluate whether loss is lower.
        # - If lower, save state dict and reset number of bad epochs
        # - If equal or higher, increase bad epochs with one
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(
                state_dict_output_dir_path,
                f'{str(dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}.pt'
            )
            torch.save(net.state_dict(), best_model_path)
            n_worse_epochs = 0
        else:
            n_worse_epochs += 1

        # Get learning rate (might have changed during training)
        lr = optimizer.param_groups[0]['lr']

        # Add loss to list
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Epoch print message
        print_epoch_message(epoch, lr, train_loss, train_mae, val_loss, val_mae, best_loss, n_worse_epochs)

    return best_model_path, train_loss_list, val_loss_list


if __name__ == "__main__":
    # Define command line options
    parser = argparse.ArgumentParser(
        prog='Train neural network',
        description='This program trains a neural network'
    )
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='N epochs without loss reduction before LR reduction')
    parser.add_argument('--df_path', type=str, required=True, help='Path to TSV with id, image path and label columns')
    parser.add_argument('--output_root_path', type=str, required=True, help='Path to directory to write output to')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.1, help='Factor by which to reduce LR on Plateau')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data loaders')
    parser.add_argument('--column_name_id', type=str, default='subject_id', help='Column name of the id column')
    parser.add_argument('--column_name_img_path', type=str, default='img_path', help='Column name of image path column')
    parser.add_argument('--column_name_label', type=str, default='subject_age', help='Column name of the label column')
    parser.add_argument('--train_fraction', type=float, default=0.6, help='Fraction of data for training')
    parser.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--state_dict_path', type=str, default=None, help='Specify path to state_dict if desired')
    parser.add_argument('--n_subjects', type=int, default=None, help='Number of participants in total dataset')
    parser.add_argument('--transfer_learning', action='store_true', help='Only update FC layer?')

    args = parser.parse_args()

    # Split data and get data loaders
    df = pd.read_csv(args.df_path, sep='\t')
    if args.n_subjects is not None:
        df = df.sample(frac=1).reset_index(drop=True)  # Source: https://stackoverflow.com/questions/29576430
        df = df.iloc[:args.n_subjects]
    colnames_dict = {'id': args.column_name_id, 'img_path': args.column_name_img_path, 'label': args.column_name_label}
    train_df, val_df, test_df = split_data(df, colnames_dict, args.output_root_path,
                                           args.train_fraction, args.val_fraction, args.test_fraction)
    train_loader = get_data_loader(train_df, 'train', colnames_dict, batch_size=args.batch_size)
    val_loader = get_data_loader(val_df, 'validation', colnames_dict, batch_size=args.batch_size)

    # Load network
    net = DenseNet(3, 1, 1)
    net = get_weights(net, args.state_dict_path)
    if args.transfer_learning:
        # Freeze all weights in the network
        for param in net.parameters():
            param.requires_grad = False
        # Unfreeze weights of the fully connected layer
        for param in net.class_layers.out.parameters():
            param.requires_grad = True

    # Set optimizer, scheduler and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    criterion = nn.L1Loss()

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create state_dict folder in output folder
    state_dict_output_dir_path = os.path.join(args.output_root_path, 'state_dicts')
    if not os.path.exists(state_dict_output_dir_path):
        os.mkdir(state_dict_output_dir_path)

    # Train
    print('**BEGINNING TRAINING***')
    best_model_path = train(args.n_epochs, device, train_loader, val_loader, optimizer, net, criterion,
                            scheduler, state_dict_output_dir_path)
