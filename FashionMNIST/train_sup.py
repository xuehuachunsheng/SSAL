# 监督学习
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from losses import S4L
import unsupresnet
from torch.autograd import Variable
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import argparse

from ssal import SelfSupervisedAL
from dataset import FashionMNISTDataset,FashionMNISTDatasetAL

parser = argparse.ArgumentParser(description="CIFAR10 SSAL Running")
parser.add_argument('--gpu', default=0, type=int, help="gpu id")

# SSAL parameters
parser.add_argument('--q_budget', default=2000, type=int)
parser.add_argument('--dc', default=0.05, type=float)
parser.add_argument('--info', default="F", type=str)
parser.add_argument('--balance', default="True", type=str)

# Training parameters
parser.add_argument('--bs', default=64, type=int, help="batch size")
parser.add_argument('--E1', default=6, type=int)
parser.add_argument('--E2', default=20, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

# 设置全局参数
BATCH_SIZE = args.bs
E1 = args.E1
E2 = args.E2 # 训练EPOCH数量 = E1xE2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_file = "/home/wyx/vscode_projects/SSAL/models/FashionMNIST/{}-{}/sup/train_loss.csv".format(args.info, args.balance)
val_loss_file = "/home/wyx/vscode_projects/SSAL/models/FashionMNIST/{}-{}/sup/val_loss.csv".format(args.info, args.balance)
test_acc_file = "/home/wyx/vscode_projects/SSAL/models/FashionMNIST/{}-{}/sup/test_acc.csv".format(args.info, args.balance)
states_file_path = "/home/wyx/vscode_projects/SSAL/models/FashionMNIST/{}-{}/sup/states.json".format(args.info, args.balance)
best_model_path = "/home/wyx/vscode_projects/SSAL/models/FashionMNIST/{}-{}/sup/best_model.pth".format(args.info, args.balance)

if not os.path.exists(os.path.dirname(train_loss_file)):
    os.makedirs(os.path.dirname(train_loss_file))
    
with open(train_loss_file, "w") as f:
    f.write("Epoch,Loss\n")
with open(val_loss_file, "w") as f:
    f.write("Epoch,Loss\n")
with open(test_acc_file, "w") as f:
    f.write("Epoch,ACC\n")

# 读取数据
dataset_L, dataset_U = FashionMNISTDatasetAL.createLURandomly(nL=500, image_size=32)
train_loader = torch.utils.data.DataLoader(dataset_L, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
dataset_test = FashionMNISTDataset(train=False, image_size=32)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

n_classes = FashionMNISTDataset.C # total class number

# loss 类型
criterion = S4L(w=1)

#model = models.resnet18(pretrained=False)
#model.conv1 = nn.Conv2d(3, model.inplanes, kernel_size=3, stride=2, padding=3, bias=False)
#model.fc = nn.Linear(512, n_classes)
# 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
#model.fc.bias.data.fill_(-np.log(n_classes-1))

# Using the pytorch-cifar10 implementation
# The 1st convolution layer for CIFAR-10 is 3x3
model = unsupresnet.ResNet(unsupresnet.BasicBlock, [2, 2, 2, 2], num_classes=n_classes)
model.to(DEVICE)

ssal = SelfSupervisedAL(L=dataset_L,
                        U=dataset_U,
                        C=n_classes,
                        feature_extractor=model,
                        classifier=model.sup_linear,
                        q_budget=args.q_budget,
                        DEVICE=DEVICE,
                        dc=args.dc,
                        info_metric=args.info,
                        balance=eval(args.balance))

#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 定义训练过程
def train(epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print("NumSamples:", total_num, "NumBatch: ", len(train_loader))
    start = time.time()
    for i, (L_data,L_target) in enumerate(train_loader):
        L_data, L_target = Variable(L_data).to(DEVICE), Variable(L_target).to(DEVICE)
        sup_out = model(L_data, sup=True)
        loss = criterion.loss(sup_out, L_target, None, None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        if (i + 1) % 500 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tTime: {:.2f} s'.format(
                epoch, (i + 1) * BATCH_SIZE, total_num,
                       100. * (i + 1) / len(train_loader), loss.item(), time.time() - start),end="",flush=True) 
            start = time.time()
            
    ave_loss = sum_loss / len(train_loader)
    print('\nEpoch:{}, Loss:{:.4f}'.format(epoch, ave_loss))
    with open(train_loss_file, "a") as f:
        f.write("{},{:.4f}\n".format(epoch, ave_loss))

def val(epoch):
    model.eval()
    sum_loss = 0
    real_labels = []
    predict_labels = []
    total_num = len(test_loader.dataset)
    print("Test samples Num: {}, Batch Num: {}".format(total_num, len(test_loader)))
    start = time.time()
    with torch.no_grad():
        for i, (L_data, L_target) in enumerate(test_loader):
            L_data, L_target = Variable(L_data).to(DEVICE), Variable(L_target).to(DEVICE)
            sup_out = model(L_data, sup=True)
            _, pred = torch.max(sup_out.data, dim=1)
            _, real = torch.max(L_target, dim=1)
            real_labels.extend(list(real.cpu().numpy()))
            predict_labels.extend(list(pred.cpu().numpy()))
            loss = criterion.loss(sup_out, L_target,None,None)
            sum_loss += loss.data.item()
            if (i + 1) % 100 == 0:
                print('\rTest Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tTime: {:.2f} s'.format(
                    epoch, (i + 1) * BATCH_SIZE, total_num,
                        100. * (i + 1) / len(test_loader), loss.item(), time.time() - start),end="",flush=True) 
                start = time.time()
            
    ave_loss = sum_loss / len(test_loader)
    print('\nEpoch:{}, Loss:{:.4f}'.format(epoch, ave_loss, ave_loss))
    with open(val_loss_file, "a") as f:
        f.write("{},{:.4f}\n".format(epoch, ave_loss))
    
    assert len(real_labels) == total_num
    acc = 0
    for i in range(len(real_labels)):
        if int(real_labels[i]) == int(predict_labels[i]):
            acc += 1
    acc /= total_num
    print('Epoch: {}, ACC: {:.2f}%'.format(epoch, 100 * acc))
    with open(test_acc_file, "a") as f:
        f.write("{},{:.4f}\n".format(epoch,acc))
    return acc

# Training process
#rho_delta_list = []
best_acc = 0
for epoch in range(1, E1*E2 + 1):
    print("Training....")
    train(epoch)
    print("Testing....")
    acc = val(epoch)
    if np.mean(acc) > best_acc:
        best_acc = np.mean(acc)
        torch.save(model.state_dict(), best_model_path)
    
    if epoch % E2 == 0:
        print("AL Beginning.....")
        # Load the best model
        model.load_state_dict(torch.load(best_model_path))
        # Select samples and updating
        model.eval()
        ssal.select()
        ssal.store_states(states_file_path)
        ssal.update_data()
        train_loader = torch.utils.data.DataLoader(ssal.L, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
        print("AL End.....")
        