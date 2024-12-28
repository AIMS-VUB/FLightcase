# Inspiration: https://github.com/OpenMRBenelux/openmr2021-dataviz-workshop/blob/main/binder/postBuild
# Create conda environment FLightcase
conda env create --file environment.yml

conda activate FLightcase

# Get data
datalad clone https://github.com/OpenNeuroDatasets/ds003083.git inputs/ds003083/
datalad clone https://github.com/OpenNeuroDatasets/ds000229.git inputs/ds000229/
datalad clone https://github.com/OpenNeuroDatasets/ds005530.git inputs/ds005530/

datalad get inputs/ds003083/*/anat/*T1w.nii.gz
datalad get inputs/ds000229/*/anat/*T1w.nii.gz
datalad get inputs/ds005530/*/anat/*T1w.nii.gz

pip3 install ..