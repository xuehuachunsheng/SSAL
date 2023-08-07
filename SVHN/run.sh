echo "Running SVHN"
cd /home/wyx/vscode_projects/SSAL/SVHN
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/SVHN/train.py \
    --gpu 0 \
    --info F \
    --balance True \
    > log.out 2>&1 &

cd /home/wyx/vscode_projects/SSAL/SVHN
nohup /home/wyx/env/python388/bin/python3 -u /home/wyx/vscode_projects/SSAL/SVHN/train_sup.py \
    --gpu 1 \
    --info F \
    --balance True \
    > log.out 2>&1 &
