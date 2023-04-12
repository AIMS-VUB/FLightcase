"""
Python implementation of "ANOVA from intermediary results", Anders et al. 2016:
==> https://www.biochemia-medica.com/en/journal/27/2/10.11613/BM.2017.026

This program can be checked with:
==> https://statpages.info/anova1sm.html
"""

import argparse
import numpy as np
from scipy.stats import f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = None, help='Path to output txt file')
    parser.add_argument("--n_list", nargs="+", help="List sample sizes. CAVE: order should match other arguments.")
    parser.add_argument("--avg_list", nargs="+", help="List averages. CAVE: order should match other arguments.")
    parser.add_argument("--sd_list", nargs="+", help="List standard deviations. CAVE: order should match other arguments.")
    args = parser.parse_args()

    n_list = [float(i) for i in args.n_list]
    avg_list = [float(i) for i in args.avg_list]
    sd_list = [float(i) for i in args.sd_list]

    # Sum of squares
    ssb = 0             # Sum of squares between groups
    ssw = 0             # Sum of squares within groups
    avg_all_observations = sum([n * avg for n, avg in zip(n_list, avg_list)])/sum(n_list)

    for n, avg, sd in zip(n_list, avg_list, sd_list):
        ssb += n * (avg - avg_all_observations) ** 2
        ssw += (n-1) * sd ** 2

    sst = ssw + ssb     # Total sum of squares

    # Degrees of freedom
    dfb = len(n_list) - 1
    dfw = sum(n_list) - len(n_list)

    # Mean square
    msb = ssb/dfb
    msw = ssw/dfw

    # Calculate F
    # Additional source: https://blog.minitab.com/en/adventures-in-statistics-2/understanding-analysis-of-variance-anova-and-the-f-test
    F = msb/msw

    # Calculate p
    # Additional source: https://stackoverflow.com/questions/30165674/am-looking-for-an-equivalent-function-for-fdist-in-python
    p = 1-f.cdf(F, dfb, dfw)

    text = f'==============\n'\
           f'ANOVA results:\n'\
           f'==============\n\n'\
           f'INPUT:\n'\
           f'- n list: {n_list}\n'\
           f'- avg list: {avg_list}\n'\
           f'- sd list: {sd_list}\n\n'\
           f'OUTPUT:\n'\
           f'- Sum of squares between: {ssb}\n'\
           f'- Sum of squares within: {ssw}\n'\
           f'- Sum of squares total: {sst}\n'\
           f'- Degrees of freedom between: {dfb}\n'\
           f'- Degrees of freedom within: {dfw}\n'\
           f'- Mean square between: {msb}\n'\
           f'- Mean square within: {msw}\n'\
           f'- F: {F}\n'\
           f'- p: {p}'

    # Save in .txt file if path is provided
    if args.output_path is not None:
        with open(args.output_path, 'w') as txt_file:
            txt_file.write(text)

    # Print
    print(text)
