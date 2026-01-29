# Inspiration: https://github.com/OpenMRBenelux/openmr2021-dataviz-workshop/blob/main/binder/postBuild

# Get script directory path of this file and cd
# From: https://medium.com/@forest.dewberry/bash-from-within-a-script-refer-to-the-scripts-parent-directory-2eb10fab5b13
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"
cd "${SCRIPT_DIR}" || exit
echo
echo ">>> Switched to working directory: $(pwd) <<<"
echo

## Preprocess data
echo
echo "######################"
echo "Preprocessing data ..."
echo "######################"
echo
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds003083/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds000229/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
python3 preprocess_images.py --dataset_root_path "${SCRIPT_DIR}/inputs/ds005530/" --preprocessing_name Wood_2022_downsampled --output_resolution 26 26 26
