# FLightcase example in Binder
This subdirectory contains a local federated learning (FL) simulation using FLightcase, 
fine-tuning an existing [brain age model](https://github.com/MIDIconsortium/BrainAge/blob/46800008b9ed79551988230f2f5470f8cf0a9ead/Models/T1/Skull_stripped/seed_60.pt) on healthy control MRI data from [OpenNeuro](https://openneuro.org/).
The simulation is hosted in [Binder](https://mybinder.readthedocs.io/en/latest/index.html), and 
consists of 1 server node and 3 client nodes, in analogy to the real-world example in [this preprint](https://www.medrxiv.org/content/10.1101/2023.04.22.23288741v1).
Each node is represented by a separate FL workspace, a separate directory.

| client     | OpenNeuro dataset                                                  | n_subjects | n_sessions_per_subject |
|------------|--------------------------------------------------------------------|------------|------------------------|
| Brussels   | [ds003083](https://openneuro.org/datasets/ds003083/versions/1.0.1) | 26         | 1                      |
| Greifswald | [ds000229](https://openneuro.org/datasets/ds000229/versions/00001) | 15         | 1                      |
| Prague     | [ds005530](https://openneuro.org/datasets/ds005530/versions/1.0.8) | 18         | 1                      |

Click the badge below to get started!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIMS-VUB/FLightcase/HEAD)
