# Inspiration: https://github.com/OpenMRBenelux/openmr2021-dataviz-workshop/blob/main/binder/postBuild
# Note: please make sure to create a Conda environment and to activate it before running this bash script
# - conda env create --file environment.yml
# - conda activate FLightcase_sim_data_prep

# Get script directory path of this file and cd
# From: https://medium.com/@forest.dewberry/bash-from-within-a-script-refer-to-the-scripts-parent-directory-2eb10fab5b13
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"
cd "${SCRIPT_DIR}" || exit
echo
echo ">>> Switched to working directory: $(pwd) <<<"
echo

## Get data
echo
echo "####################"
echo "Downloading data ..."
echo "####################"
echo

mkdir inputs
aws s3 sync --no-sign-request s3://openneuro.org/ds000229 inputs/ds000229 --exclude "*" --include "participants.tsv" --include "*T1w.nii.gz"
aws s3 sync --no-sign-request s3://openneuro.org/ds003083 inputs/ds003083 --exclude "*" --include "participants.tsv" --include "*T1w.nii.gz"
aws s3 sync --no-sign-request s3://openneuro.org/ds005530 inputs/ds005530 --exclude "*" --include "participants.tsv" --include "*T1w.nii.gz"

## Preprocess data
echo
echo "######################"
echo "Preprocessing data ..."
echo "######################"
echo
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds003083/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds000229/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds005530/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
