"""
Template for defining your own network architecture.
Note: the variable "net_architecture" must be defined
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet

###########
# Example 1
###########
# Type: single input network with variable trainable parameters


class T1Net(nn.Module):
    def __init__(self) -> None:
        super(T1Net, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8000000, out_features=2).requires_grad_(True),
            nn.Linear(in_features=2, out_features=1).requires_grad_(False)
        )

    def forward(self, t1_img_tensor):
        latent = self.flatten(t1_img_tensor)
        x = self.fc(latent)
        return x


net_architecture = T1Net()

###########
# Example 2
###########
# Type: Double input network
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
