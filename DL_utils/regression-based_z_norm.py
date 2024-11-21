import os
import re
import BICAMSZ
import argparse
import numpy as np
import pandas as pd


# Z-normalise cognition
# --> BRUSSELS
def z_normalise_sdmt_brussels(df):
    df.drop(['SDMT_norm', 'SDMT_z', 'SDMT_z-norm'], axis=1)  # Remove old z-normalisation columns
    df['sex_num'] = df['Gender'].apply(lambda x: {'M': 1, 'F': 2}.get(x))
    df['Age_lower'] = df['Age'].apply(lambda x: np.floor(x)).astype(int)
    z_cutoff = -1
    input_columns = ['Age_lower', 'sex_num', 'Education', 'SDMT_Raw']
    new_columns = [f'SDMT_regr_z', f'SDMT_regr_z_imp']
    df[new_columns] = df[input_columns].apply(BICAMSZ.pipeline_for_pandas, args=('sdmt', z_cutoff), axis=1)

    return df


# --> GREIFSWALD
def _gw_predict_sdmt(age, sex, edu):
    return 10.1914 + 0.1080 * sex + 0.0076 * age - 0.0012 * age**2 + 0.4894 * edu


def _gw_raw_to_scaled(raw):
    cutoffs = [26, 29, 33, 39, 43, 48, 52, 56, 60, 64, 67, 72, 76, 78, 81, 90, 102, 108]
    for i, cutoff in enumerate(cutoffs):
        i += 2  # Start from 2
        if raw <= cutoff:
            return i
        elif cutoff == cutoffs[-1]:
            return i + 1


def gw_z_normalisation(sdmt, age, sex, edu):
    sdmt_scaled = _gw_raw_to_scaled(sdmt)
    sdmt_pred = _gw_predict_sdmt(age, sex, edu)

    return (sdmt_scaled - sdmt_pred)/1.9087


def gw_z_norm_pipeline_for_pandas(row):
    return pd.Series(gw_z_normalisation(row[0], row[1], row[2], row[3]))


def gw_encode_education_level(edu_years):
    if edu_years <= 8:
        return 1
    elif edu_years == 9:
        return 2
    elif edu_years in [10, 11]:
        return 3
    elif edu_years >= 12:
        return 4
    elif np.isnan(edu_years):
        return np.nan
    else:
        raise ValueError('Edu years not understood')


def z_normalise_sdmt_greifswald(df):
    # Z-normalisation
    df['sex_0_1'] = df['patient_sex'].apply(lambda x: {'F': 0, 'M': 1}.get(x))
    df['edu_dummy'] = df['educational years'].apply(gw_encode_education_level)

    input_columns_sdmt = ['SDMT', 'Age', 'sex_0_1', 'edu_dummy']  # CAVE: Correct order!

    df['SDMT_regr_z'] = df[input_columns_sdmt].apply(lambda x: gw_z_norm_pipeline_for_pandas(x), axis=1)
    df['SDMT_regr_z_imp'] = df['SDMT_regr_z'].apply(lambda x: int(x <= -1))  # Binarise

    return df


# --> Prague
def prg_clean_numeric_col(x):
    if type(x) == str:
        if re.findall('\A[0-9]+,[0-9]+\Z', x):
            return float(x.replace(',', '.'))
        elif re.findall('\A[0-9]+\Z', x):
            return float(x)
    elif type(x) in [float, int]:
        return x
    else:
        ValueError(f'Edu years not converted: {x}')


def prg_to_edu_dummy(x):
    if not np.isnan(x) and type(x) in [float, int]:
        if x >= 16:
            return 1
        else:
            return 0
    else:
        return np.nan


def prg_regression_based_z_normalisation(age, edu_dummy, sdmt_90):
    return (sdmt_90
            - 67.18042161
            + 0.0002014472924 * age**3
            - 0.000002358544643 * age**4
            - 3.864964401*edu_dummy) \
            / -8.342252676


def z_normalise_sdmt_prague(df):
    # df = pd.read_csv(args.input_path_tsv, sep='\t')
    age_colname = 'age_at_scan'  # Appears to be more or less equal to 'age_psy_assessment_calc'
    edu_years_colname = 'edu_years'
    sdmt_colname = 'sdmt90_total'

    # Clean edu_years and age_psy_assessment_calc columns
    df[edu_years_colname] = df[edu_years_colname].apply(prg_clean_numeric_col)
    df[age_colname] = df[age_colname].apply(prg_clean_numeric_col)

    # Create dummy variable for education level
    df['edu_dummy'] = df[edu_years_colname].apply(prg_to_edu_dummy)

    # Calculate z-score
    df['SDMT_regr_z'] = df.apply(lambda x: prg_regression_based_z_normalisation(x[age_colname], x['edu_dummy'], x[sdmt_colname]), axis=1)
    df['SDMT_regr_z_imp'] = df['SDMT_regr_z'].apply(lambda x: int(x <= -1))  # Binarise

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", required=True, help="Path to TSV file")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument('--center', required=True, help="Which center? "
                                                        "Choose from ['Brussels', 'Greifswald', 'Prague']")
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.tsv_path, sep='\t')

    # Add z-normalisation
    if args.center == 'Brussels':
        df_with_z_and_imp = z_normalise_sdmt_brussels(df)
    elif args.center == 'Greifswald':
        df_with_z_and_imp = z_normalise_sdmt_greifswald(df)
    elif args.center == 'Prague':
        df_with_z_and_imp = z_normalise_sdmt_prague(df)
    else:
        raise ValueError(f'Center {args.center} not recognized')

    # Save to TSV file
    df_with_z_and_imp.to_csv(os.path.join(args.output_dir, "participants.tsv"), index=False, sep='\t')
