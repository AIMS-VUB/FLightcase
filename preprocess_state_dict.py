"""
This script removes the "module" keyword from the keys in a state dict
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import os
import sys
# Source: https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'submodules', 'Wood_2022'))
import torch
import argparse
from submodules.Wood_2022.fine_tune import convert_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to state dict to be processed')
    parser.add_argument('--output_path', type=str, help='Path to write processed state dict to')
    args = parser.parse_args()

    new_state_dict = convert_state_dict(args.input_path)
    torch.save(new_state_dict, args.output_path)
