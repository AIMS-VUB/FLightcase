"""
Create a state dict of a net initialised with Glorot initialisation

Initialise with Glorot initialisation (Xavier uniform)
- https://pytorch.org/docs/stable/nn.init.html
- https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
"""

import os
import argparse
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet

def init_weights(m):
    """
    Function from: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    """
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", help='Specify output path to store state dict in', type=str)
    args = parser.parse_args()

    # Initialise DenseNet
    net = DenseNet(3,1,1)
    net.apply(init_weights)

    # Save State Dict
    model_path = os.path.join(args.output_dir_path, 'Glorot_weight_network.pt')
    torch.save(net.state_dict(), model_path)
