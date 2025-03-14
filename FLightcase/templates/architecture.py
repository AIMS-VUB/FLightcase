"""
Template for defining your own network architecture.

Notes:
- The variable "net_architecture" must be defined
- If you would like to freeze certain layers, add ".requires_grad_(False)" after layer
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet

###########
# Example 1
###########
# Note: Lightweight network used for the simulation


class AdaptedSFCN(nn.Module):
    """
    This is an adapted version of the Simple Fully Convolutional Network (SFCN) by Dr. Han Peng and colleagues
    The code was adapted from the "sfcn.py" file in their GitHub repository:
    ==> Link: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/model_files/sfcn.py
    ==> Link paper: https://www.sciencedirect.com/science/article/pii/S1361841520302358
    """
    def __init__(self):
        super(AdaptedSFCN, self).__init__()

        # Feature extractor
        self.feat_ext = nn.Sequential()
        self.feat_ext.conv_1 = self.conv_layer(in_channel=1, out_channel=32, maxpool=True, kernel_size=3, padding=1)
        self.feat_ext.conv_2 = self.conv_layer(in_channel=32, out_channel=64, maxpool=True, kernel_size=3, padding=1)
        self.feat_ext.conv_3 = self.conv_layer(in_channel=64, out_channel=128, maxpool=False, kernel_size=1, padding=0)

        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.average_pool = nn.AvgPool3d(kernel_size=(6, 6, 6))
        self.classifier.dropout = nn.Dropout(p=0.5)
        self.classifier.flatten = nn.Flatten()
        self.classifier.linear = nn.Linear(in_features=128, out_features=1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x_f = self.feat_ext(x)
        x = self.classifier(x_f)
        return x


net_architecture = AdaptedSFCN()


###########
# Example 2
###########
# Type: Double input network with variable trainable parameters
# CAVE: the argument order in the forward hook must match the order in the "modalities_dict" attribute
#       in the client settings JSON


class T1FlairNet(nn.Module):
    def __init__(self) -> None:
        super(T1FlairNet, self).__init__()
        self.flatten_t1 = nn.Flatten(start_dim=1, end_dim=-1)
        self.flatten_flair = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=16000000, out_features=1)

    def forward(self, t1_img_tensor, flair_img_tensor):
        latent_t1 = self.flatten_t1(t1_img_tensor)
        latent_flair = self.flatten_flair(flair_img_tensor)
        x = torch.cat((latent_t1, latent_flair), 1)
        x = self.linear(x)
        return x

# net_architecture = T1FlairNet()

###########
# Example 3
###########
# Type: Example using an imported architecture

# net_architecture = DenseNet(3, 1, 1)
