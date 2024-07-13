# FLightcase :airplane::briefcase:

This repository contains the code to perform federated learning (FL) when computers are present in the same virtual private network (VPN) and are able to share models with secure copy protocol (SCP) via secure shell (SSH).

Note: this repository was specifically written for [a first FL experiment using this set-up](https://www.medrxiv.org/content/10.1101/2023.04.22.23288741v1) [1], and is currently still specific for this approach. We aim to make it more generally applicable in the future. Furthermore, it requires datasets on client computers to be organised in a specific way, namely the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) [2]. The approach currently also requires the username on each computer to be identical.

## How to get started
A couple of preparatory steps are required to start an FL experiment:
1. Clone this repository to every computer in your FL network
2. On each computer, create a virtual environment within the cloned repository (suggested name: `venv`)
3. Install the federated learning requirements (`requirements/requirements_federated_learning.txt`) in this virtual environment (e.g. with `pip install -r requirements/requirements_federated_learning.txt` with the terminal in the FLightcase folder and after activating the virtual environment)
4. On a location of your liking on the computer, create a repository for your FL experiment. For example, name this repository `FL_experiment`. Within this repository, create a subrepository that will be the workspace on that computer (suggested name: `FL_workspace`)
5. Copy the following JSON templates to the parent experiment folder (not to the FL workspace subrepository):
   - FL_plan: `templates/FL_plan.json`
   - Settings: `templates/settings_server.json` or `templates/settings_client.json`, depending on whether the computer is the server or a client
6. In the JSON settings templates, adapt keys or values that include "[ADAPT]". Where not specifying a value is optional (null), this is indicated. In the FL plan, you can adapt the values to your preferences

To start the FL experiment, perform the following steps:
1. Make sure that all computers are present in the same virtual network via VPN
2. Run the `client.py` script on each client computer within the virtual environment
   ```sh
   python client.py --settings_path "Path/to/the/FL_settings_client.json"
3. Run the `server.py` script on the server computer within the virtual environment 
   ```sh
   python server.py --settings_path "Path/to/the/FL_settings_server.json" --FL_plan_path "Path/to/the/FL_plan.json"


That's it!

## Need help or discovered a problem?
If you experience problems with this GitHub repository, please do not hesitate to create an issue, or send a mail to [stijn.denissen@vub.be](mailto:stijn.denissen@vub.be)

## References
[1] Denissen, S., Grothe, M., Vaneckova, M., Uher, T., Laton, J., Kudrna, M., ... & Nagels, G. (2023). Transfer learning on structural brain age models to decode cognition in MS: a federated learning approach. medRxiv, 2023-04.

[2] Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P., ... & Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific data, 3(1), 1-9.
