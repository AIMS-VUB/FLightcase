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
    Additional:
    - https://stackoverflow.com/questions/21095654/what-is-a-nonetype-object
    -
    Which layers no initialisation?
    > DenseNet, ReLU, Sequential objects have no attribute 'weight'
    """
    # Weight initialisation
    try:
        nn.init.xavier_uniform_(m.weight)
    except AttributeError as AttErr:
        if str(AttErr).endswith("object has no attribute 'weight'"):
            pass
    except ValueError as ValErr:
        if str(ValErr) == 'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions':
            # For tensors with < 2 dimensions, initialise differently: https://stackoverflow.com/questions/76991945/
            nn.init.normal_(m.weight)

    # Bias initialisation
    try:
        nn.init.xavier_uniform_(m.bias)
        print('hey')
    except AttributeError as AttErr:
        if str(AttErr) == "'NoneType' object has no attribute 'dim'":
            pass
    except ValueError as ValErr:
        if str(ValErr) == 'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions':
            # ptrblck suggestion:
            # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/8
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", help='Specify output path to store state dict in', type=str)
    args = parser.parse_args()

    # Initialise DenseNet
    net = DenseNet(3, 1, 1)
    net.apply(init_weights)

    # Save State Dict
    model_path = os.path.join(args.output_dir_path, 'Glorot_weight_network.pt')
    torch.save(net.state_dict(), model_path)
