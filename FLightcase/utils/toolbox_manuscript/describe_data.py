"""
This script creates a table with dataset characteristics
"""

import json
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_tsv', type=str, help='Path to tsv file')
    parser.add_argument('--colname_id', type=str, help='Column name of ID')
    parser.add_argument('--colnames_describe_method', nargs='*', help='List the column names to assess with the '
                                                                      '.describe() method')
    parser.add_argument('--colnames_value_counts_method', nargs='*', help='List the column names to assess with the '
                                                                          '.value_counts() method')
    parser.add_argument('--settings_dict_path', type=str, help='Path to settings dict used for federated learning')
    parser.add_argument('--output_path_txt', type=str, help='Path to output txt file')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.path_to_tsv, sep='\t')

    # Filter data
    with open(args.settings_dict_path, 'r') as json_file:
        settings_dict = json.load(json_file)
    included_subjects = settings_dict.get('subject_ids')
    if included_subjects is not None:
        df = df[df[args.colname_id].isin(included_subjects)]

    # Return .txt file with values
    txt = f'==================\n' \
          f'.describe() method\n' \
          f'==================\n\n'
    for col in args.colnames_describe_method:
        txt += f'{df[col].describe()}\n\n'

    txt += f'======================\n' \
           f'.value_counts() method\n' \
           f'======================\n\n'
    for col in args.colnames_value_counts_method:
        txt += f'{df[col].value_counts()}\n\n'

    with open(args.output_path_txt, 'w') as txt_file:
        txt_file.write(txt)