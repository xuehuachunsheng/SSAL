echo "Running FashionMNIST"

cd /home/wyx/vscode_projects/SSAL/FashionMNIST
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/FashionMNIST/train.py \
    --gpu 1 \
    --info U \
    --balance True \
    > log.out 2>&1 &