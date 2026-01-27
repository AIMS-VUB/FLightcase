"""
This is a script for inference on a specific state dict
"""

import argparse
import os
import sys
import json
import pathlib
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
# Add path to great-great-grandparent dir of this Python file: https://stackoverflow.com/questions/3430372/
path_to_toolbox_manuscript_folder = str(pathlib.Path(__file__).parent.resolve())
path_to_root_FLightcase_folder = os.sep + os.sep.join(path_to_toolbox_manuscript_folder.split(os.sep)[1:-3]) + os.sep  # Start from 1 as splits on first sep
sys.path.append(path_to_root_FLightcase_folder)
from FLightcase.utils.deep_learning.general import get_device
from FLightcase.utils.deep_learning.data import get_data_loader
from FLightcase.utils.deep_learning.train import get_criterion
from FLightcase.utils.deep_learning.model import get_weights, import_net_architecture
from FLightcase.utils.deep_learning.evaluation import evaluate
from FLightcase.utils.results import create_test_true_pred_df, create_test_scatterplot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_path")
    args = parser.parse_args()

    # Extract settings
    with open(args.settings_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    modalities_dict = settings_dict.get('modalities_to_include')        # Modalities (e.g. {'anat': ['T1w', 'FLAIR']})
    colnames_dict = settings_dict.get('colnames_dict')                  # Colnames dict
    criterion_txt = settings_dict.get('criterion')                      # Criterion in txt format, lowercase (e.g. l1loss)
    batch_size = settings_dict.get('batch_size')                        # Batch size
    architecture_path = settings_dict.get('architecture_path')          # Architecture_path
    test_data_path = settings_dict.get('test_data_path')                # Test data path
    state_dict_path = settings_dict.get('state_dict_path')              # Path to state dict
    output_dir_path = settings_dict.get('output_dir_path')              # Output_dir_path
    device = get_device(settings_dict.get('device'))                    # Device

    # Load Net
    net_architecture = import_net_architecture(architecture_path)
    net = get_weights(net_architecture, state_dict_path)

    # Load test data
    test_df = pd.read_csv(test_data_path, sep='\t')
    test_loader, n_test = get_data_loader(test_df, 'test', colnames_dict, batch_size, return_n=True)

    # Test
    print('Testing...')
    criterion = get_criterion(criterion_txt)
    test_loss, true_labels_test, pred_labels_test, id_list_test = evaluate(net, test_loader, criterion, device, 'test')

    # Test result analysis
    true_pred_test_df = create_test_true_pred_df(id_list_test, true_labels_test, pred_labels_test, output_dir_path, save=True)
    create_test_scatterplot(true_pred_test_df, 'centralised', output_dir_path)
    with open(os.path.join(output_dir_path, 'test_results.csv'), 'w') as f:
        f.write(f'MAE: {mean_absolute_error(true_labels_test, pred_labels_test)}\n'
                f'Pearsonr: {pearsonr(true_labels_test, pred_labels_test)}')
