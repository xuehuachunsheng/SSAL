# SSAL
The proposed self-supervised active learning approach

## Usage 
Please create  `models/CIFAR10`, `models/CIFAR100`, `models/FashionMNIST`, `models/SVHN`, `models/TinyImageNet` folders to store the training model and the middle results.
The middle results consist of `queried sample idx`, `rho`, `delta`, `features`, `logits`, `The number of samples of each class`, `The number of queried samples of each class` in each AL round.

Please Run `train.py` or `run.sh` in each folder.
