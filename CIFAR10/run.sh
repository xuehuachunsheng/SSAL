echo "Running CIFAR10"

cd /home/wyx/vscode_projects/SSAL/CIFAR10
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/CIFAR10/train.py \
    --gpu 1 \
    --info U \
    --balance True \
    > log.out 2>&1 &

