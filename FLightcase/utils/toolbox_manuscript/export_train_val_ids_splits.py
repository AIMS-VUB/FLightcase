import os
import re
import json
import argparse
import pandas as pd


def to_json(dict_to_save, output_path):
    with open(output_path, 'w') as f:
        json.dump(dict_to_save, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_directory_dfs")
    parser.add_argument("--path_output_dir")
    parser.add_argument("--colname_id")
    parser.add_argument("--client_name")
    args = parser.parse_args()

    # Initialisations
    train_splits_dict = {}
    val_splits_dict = {}

    # Define regexps
    regexp_train = r'train_df_rs_\d+\.tsv'
    regexp_val = r'val_df_rs_\d+\.tsv'

    for element in os.listdir(args.path_directory_dfs):
        train_match = re.findall(regexp_train, element)
        val_match = re.findall(regexp_val, element)

        if bool(train_match):
            random_state = int(train_match[0].split('_')[-1].removesuffix('.tsv'))
            train_df = pd.read_csv(os.path.join(args.path_directory_dfs, train_match[0]), sep='\t')
            train_splits_dict[random_state] = list(train_df[args.colname_id])

        elif bool(val_match):
            random_state = int(val_match[0].split('_')[-1].removesuffix('.tsv'))
            val_df = pd.read_csv(os.path.join(args.path_directory_dfs, val_match[0]), sep='\t')
            val_splits_dict[random_state] = list(val_df[args.colname_id])

    # Sort dicts
    train_splits_dict_ordered = dict(sorted(train_splits_dict.items()))
    val_splits_dict_ordered = dict(sorted(val_splits_dict.items()))

    # Save to json
    to_json(train_splits_dict_ordered, os.path.join(args.path_output_dir, f'train_splits_{args.client_name}.json'))
    to_json(val_splits_dict_ordered, os.path.join(args.path_output_dir, f'val_splits_{args.client_name}.json'))
