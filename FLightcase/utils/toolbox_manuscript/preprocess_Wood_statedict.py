"""
This script removes the "module" keyword from the keys in a state dict
==> Rewritten from the "fine_tune.py" file in the GitHub repository of Wood et al. 2022
    - GitHub repo: https://github.com/MIDIconsortium/BrainAge
    - File path: https://github.com/MIDIconsortium/BrainAge/blob/main/fine_tune.py
"""

import torch
import argparse
from collections import OrderedDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to state dict to be processed')
    parser.add_argument('--output_path', type=str, help='Path to write processed state dict to')
    args = parser.parse_args()

    state_dict_old = torch.load(args.input_path, map_location='cpu')
    state_dict_new = OrderedDict()
    for module_name, module in state_dict_old.items():
        if module_name.startswith('module.'):
            state_dict_new[module_name.removeprefix('module.')] = module
        else:
            state_dict_new[module_name] = module

    torch.save(state_dict_new, args.output_path)
