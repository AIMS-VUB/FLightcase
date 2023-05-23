"""
Perform chi-squared test
"""

import argparse
import pandas as pd
import scipy.stats as stats
import numpy as np

if __name__ == "__main__":
    # https://stackoverflow.com/questions/53712889/python-argparse-with-list-of-lists
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = None, help='Path to output txt file')
    parser.add_argument("--groups", nargs="+", help="List names of the groups")
    parser.add_argument("--categories", nargs="+", help="List names of the categories")
    parser.add_argument("--counts", type=float, nargs="+", action="append",
                        help="List category counts per group, repeating --counts flag for each group, e.g.: "
                             "--counts category_1 category_2 --counts category_1 category_2")
    args = parser.parse_args()

    # Create Pandas dataframe
    df = pd.DataFrame(args.counts)
    df.index = args.groups
    df.columns = args.categories

    # Perform chi-squared test
    chi2, p, _, _ = stats.chi2_contingency(df)

    text = f'=====\n' \
           f'Input\n' \
           f'=====\n' \
           f'{df}\n\n' \
           f'=======\n' \
           f'Results\n' \
           f'=======\n\n' \
           f'- chi-squared test statistic: {chi2}\n' \
           f'- p value: {p}'

    # Save in .txt file if path is provided
    if args.output_path is not None:
        with open(args.output_path, 'w') as txt_file:
            txt_file.write(text)

    # Print
    print(text)
