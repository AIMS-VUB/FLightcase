# FLightcase simulation
:construction::wrench: Under construction :wrench::construction:

This subdirectory contains a local federated learning (FL) simulation using FLightcase, 
fine-tuning a convolutional neural network (CNN) on healthy control MRI data from [OpenNeuro](https://openneuro.org/).
The simulation consists of 1 server node and 3 client nodes, in analogy to the real-world example in [this preprint](https://www.medrxiv.org/content/10.1101/2023.04.22.23288741v1) [1].
Each node is represented by a separate FL workspace, a separate directory.

| client name | OpenNeuro dataset                                                  | n_subjects | n_sessions_per_subject |
|-------------|--------------------------------------------------------------------|------------|------------------------|
| client_1    | [ds003083](https://openneuro.org/datasets/ds003083/versions/1.0.1) | 26         | 1                      |
| client_2    | [ds000229](https://openneuro.org/datasets/ds000229/versions/00001) | 15         | 1                      |
| client_3    | [ds005530](https://openneuro.org/datasets/ds005530/versions/1.0.8) | 18         | 1                      |

Note: Due to a persistent SSHException during local simulation on Mac, for local simulation using 127.0.0.1 as IP address, "cp" is used instead of scp to copy files.

***

## Requirements
This simulation requires cloning the GitHub repository rather than pip installing FLightcase.\
Furthermore, besides the requirements listed in the parent directory README, the preparation of this simulation relies on conda.\
I would recommend installing miniconda. Click [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for more info.

***

## How to get started?
### Clone this GitHub repository
```git clone https://github.com/AIMS-VUB/FLightcase.git```

### Change directory to the GitHub repository
```cd FLightcase```

### Get submodule contents
Get the contents of the BrainAge submodule ([MIDIconsortium "BrainAge" GitHub repo](https://github.com/MIDIconsortium/BrainAge)).
1. ```git submodule init``` (initialise local configuration file)
2. ```git submodule update``` (get the project contents)

Note: Click [this link](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more info about this process.

### Download and preprocess the data
The data will be available in an "inputs" folder within the "simulation" folder
1. Navigate in the terminal to the "simulation" subfolder: ```cd FLightcase/simulation/```
2. ```conda env create --file environment.yml``` (creates conda environment "FLightcase_sim_data_prep")
3. ```conda activate FLightcase_sim_data_prep``` (activates "FLightcase_sim_data_prep" environment)
4. ```bash prepare_data.sh``` (downloads and preprocesses data (pipeline by Wood et al. 2022 [2]))
5. ```conda deactivate``` (deactivates conda environment)
6. Return to parent "FLightcase" directory: ```cd ../..```

#### Note:
On 9 January 2025, we obtained permission by the authors of the image preprocessing pipeline [2] to include an adapted version of the [pre_process.py file](https://github.com/MIDIconsortium/BrainAge/blob/main/pre_process.py) from their [GitHub repository](https://github.com/MIDIconsortium/BrainAge) (also included in this simulation as submodule) in ours.
The file was adapted to work with newer Python and dependency versions. We sincerely thank Dr. Thomas Booth, Dr. David Wood and Dr. Onur Ülgen for allowing us to do so!

The adaptations:
- Replacing the "AddChannel" Monai class with "EnsureChannelFirst", as the former was [deprecated since Monai version 0.8 and removed in version 1.3](https://docs.monai.io/projects/monai-deploy-app-sdk/en/0.6.0/notebooks/tutorials/02_mednist_app.html). We also replaced the "reoriented_arr" input with "reoriented_arr.copy()".
- Removing the "-mode" flag (set to "fast") when running the hd-bet CLI
- In the spacing phase using Monai's "Spacing" class, we:
  - removed the "reoriented_affine" input
  - removed the step taking the first element of the output

### Prepare virtual environment
Note: only one virtual environment needs to be created, which can be used by all virtual nodes.
1. ```python3 -m venv .FLightcase_venv```
2. ```source .FLightcase_venv/bin/activate```
3. Install FLightcase. There are 2 options:
   - Building FLightcase from this GitHub repository: ```pip3 install -r requirements.txt```
   - Downloading and installing via [PyPI](https://pypi.org/project/FLightcase/): ```pip3 install FLightcase==0.1.5``` (Adapt version number if desired)

### Prepare workspaces
For this, we refer to the eponymous header in the README in the parent directory.
- We recommend to create an "FL_simulation" parent folder, and to create the four workspace directories in this folder.
- The AdaptedSFCN network in the template "architecture.py" file is used for this simulation.
  - Note: This is an adapted version of the Simple Fully Convolutional Network (SFCN) architecture by Dr. Han Peng and colleagues
  - Adapted from the ["sfcn.py" file](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/model_files/sfcn.py) in the [UKBiobank_deep_pretrain GitHub repository](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)
  - Link to paper: click [here](https://www.sciencedirect.com/science/article/pii/S1361841520302358)
- When filling in the settings JSON files:
  - The names listed in the table above can be used.
  - As this is a local simulation, please choose "127.0.0.1" as the ip address for each node (server and clients).
  - As all data sets did not specify a session, remove the "session" key from the settings JSON per client

***

## Running FLightcase
Open 4 terminals, one for the server, and 3 for the clients. Activate the virtual environment on each terminal.
1. On client nodes, run: ```FLightcase run-client --settings_path /path/to/client_node_settings.json```
2. On the server node, run ```FLightcase run-server --settings_path /path/to/server_node_settings.json```

Enjoy the show! :woman_dancing::man_dancing:

***

## References
[1] Denissen, S., Grothe, M., Vaneckova, M., Uher, T., Laton, J., Kudrna, M., ... & Nagels, G. (2023). Transfer learning on structural brain age models to decode cognition in MS: a federated learning approach. medRxiv, 2023-04.

[2] Wood, D. A., Kafiabadi, S., Al Busaidi, A., Guilhem, E., Montvila, A., Lynch, J., ... & Booth, T. C. (2022). Accurate brain‐age models for routine clinical MRI examinations. Neuroimage, 249, 118871.
