echo "Running CIFAR100"

cd /home/wyx/vscode_projects/SSAL/CIFAR100
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/CIFAR100/train.py \
    --gpu 0 \
    --info U \
    --balance True \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/CIFAR100
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/CIFAR100/train_sup.py \
    --gpu 1 \
    --info F \
    --balance True \
    > log.out 2>&1 &