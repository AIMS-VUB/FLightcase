"""
Test script
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import torch
import argparse
import pandas as pd
import torch.nn as nn
from monai.networks.nets import DenseNet
from DL_utils.data import get_data_loader
from DL_utils.model import get_weights
from DL_utils.evaluation import evaluate


if __name__ == "__main__":
    # Define command line options
    parser = argparse.ArgumentParser(
        prog='Train neural network',
        description='This program trains a neural network'
    )
    parser.add_argument('--state_dict_path', type=str, required=True, help='Specify path to state_dict if desired')
    parser.add_argument('--test_df_path', type=str, required=True, help='Specify path to state_dict if desired')
    parser.add_argument('--column_name_id', type=str, default='subject_id', help='Column name of the id column')
    parser.add_argument('--column_name_img_path', type=str, default='img_path', help='Column name of image path column')
    parser.add_argument('--column_name_label', type=str, default='subject_age', help='Column name of the label column')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data loaders')
    args = parser.parse_args()

    # Set device and define criterion
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()

    # Get data loader
    test_df = pd.read_csv(args.test_df_path, sep='\t')
    colnames_dict = {'id': args.column_name_id, 'img_path': args.column_name_img_path, 'label': args.column_name_label}
    test_loader = get_data_loader(test_df, 'test', colnames_dict, batch_size=args.batch_size)

    # Load network
    net = DenseNet(3, 1, 1)
    net = get_weights(net)

    # Test
    val_loss, val_true_label_list, val_pred_label_list = evaluate(net, test_loader, criterion, device,
                                                                  'test', print_message=True)
