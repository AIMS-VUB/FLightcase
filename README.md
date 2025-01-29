# FLightcase :airplane::briefcase:

A federated Learning toolbox for neuro-image research, based on secure copy protocol (SCP) via secure shell (SSH).\
It was first introduced in a [preprint in medrXiv](https://www.medrxiv.org/content/10.1101/2023.04.22.23288741v1)[1], 
and now contains a Command-Line Interface (CLI): ```FLightcase``` 

![PyPI](https://img.shields.io/pypi/v/FLightcase?label=pypi%20package)

***

## Requirements
- Unix-based operating system for each node in the network
- All computers in the same network (e.g. connected via VPN), identifiable via an IP Address
- All datasets in the Brain Imaging Data Structure (BIDS)([2])
- Make sure sending files via SCP is possible (e.g. on Mac, enable "Remote Login" (click [here](https://discussions.apple.com/thread/1738370?sortBy=rank) for more info))

***

## In brief
The FLightcase toolbox works by sending files via SCP between computers. To ensure full transmission of a file, a .txt file is sent to mark transmission completion.
Each node (server and clients) prepares a "workspace", which is a local directory that collect files. Files are shared between computers by knowing each other's workspace location.
This, and other information, is available in a JSON metadata file included in each workspace directory.
The server additionally defines the Federated Learning plan in a JSON file, containing the parameters for the FL process.

***

## How to get started
### Install FLightcase
Install FLightcase in a virtual environment or conda environment on each node. Example for virtual environment:
1. ```python3 -m venv .venv``` (creates virtual environment called ".venv" in current workspace)
2. ```source .venv/bin/activate``` (activate virtual environment)
3. ```pip3 install FLightcase``` (optional: define version number, e.g. ```pip3 install FLightcase==0.1.0```)

### Preparing the workspaces
The ```FLightcase prepare-workspace``` command is used for preparing the local workspace. Define two flags:
- ```--who```: client or server
- ```--workspace_path```: path to workspace directory

Running it will first check whether the workspace path exists, and create it if this is not the case.\
Then, an instruction menu is to be completed in your terminal:
- ```--who client```: Preparation of the *client_node_settings.json* file
- ```--who server```: Preparation of the *server_node_settings.json*, the *FL_plan.json* file and the *architecture.py* file

When preparing each JSON file, there are two options:
1. Copying the template file to the workspace. The template can then be filled in manually with any preferred text editor.
2. Filling the template file in the terminal. The instruction menu will, step by step, complete the template with you. It will then be saved to the workspace.
For the server, the *architecture.py* will always be copied to the workspace without being completed in the terminal. Please update this file to your preferred network architecture.

***

## Running FLightcase
Make sure your terminal is in the virtual environment that contains FLightcase on each node. Then:
1. On client nodes, run: ```FLightcase run-client --settings_path /path/to/client_node_settings.json```
2. On the server node, run ```FLightcase run-server --settings_path /path/to/server_node_settings.json```

Enjoy the show! :woman_dancing::man_dancing:

### Note:
A file clean-up will be performed in the FL workspace after federated learning has completed.
All files except the node settings, FL plan and architecture will be moved to a subdirectory marked by date and time stamp.
The FL_plan and architecture will be copied to this subdirectory to keep a log of the exact experiment that was performed.

***

## Need help or discovered a problem?
If you experience problems with this GitHub repository, please do not hesitate to create an issue, or send a mail to [stijn.denissen@vub.be](mailto:stijn.denissen@vub.be)

***

## References
[1] Denissen, S., Grothe, M., Vaneckova, M., Uher, T., Laton, J., Kudrna, M., ... & Nagels, G. (2023). Transfer learning on structural brain age models to decode cognition in MS: a federated learning approach. medRxiv, 2023-04.

[2] Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P., ... & Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific data, 3(1), 1-9.
