import os, sys, json
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ComposeLoaderLU:
    def __init__(self, dataset_L, dataset_U, batch_size):
        self.batch_size = batch_size
        L_len = len(dataset_L)
        U_len = len(dataset_U)
        L_batch = int(np.ceil(batch_size / (4 * U_len/L_len + 1)))
        self.loader_L = DataLoader(dataset_L, batch_size=L_batch, shuffle=True, num_workers=16)
        self.loader_U = DataLoader(dataset_U, batch_size=1, shuffle=True, num_workers=16)
        print("L size/batchsize: {}/{}".format(len(dataset_L), L_batch))
        print("U size/batchsize: {}/{}".format(len(dataset_U), self.batch_size - L_batch))
        
        self.loader_L_it = iter(self.loader_L)
        self.loader_U_it = iter(self.loader_U)
        
        self.current_batch_L = 0
        self.current_batch_U = 0
        
    def __next__(self): # 该函数不控制结束，结束控制交给外部处理
        L_data, L_target = next(self.loader_L_it, (None,None))
        if L_data is None and L_target is None:
            self.loader_L_it = iter(self.loader_L)
            L_data, L_target = next(self.loader_L_it)
        assert L_data is not None and L_target is not None, "Error in ComposeLoaderLU"
        U_data = []
        U_target = []
        count_U = 0
        while count_U < self.batch_size - self.loader_L.batch_size:
            # The shape is NxCxHxW
            U_datum, _ = next(self.loader_U_it, (None, None))
            if U_datum is None:
                self.loader_U_it = iter(self.loader_U)
                U_datum, _ = next(self.loader_U_it)
            U_datum = U_datum[0,...]
            U_data.append(U_datum)
            U_target.append([1,0,0,0]) # 没有旋转，标签为0
            # 旋转图像
            U_datum_90 = torch.rot90(U_datum,k=1,dims=[1,2])
            U_data.append(U_datum_90)
            U_target.append([0,1,0,0]) # 90，标签为1
            U_datum_180 = torch.rot90(U_datum,k=2,dims=[1,2])
            U_data.append(U_datum_180)
            U_target.append([0,0,1,0]) # 180，标签为2
            U_datum_270 = torch.rot90(U_datum,k=3,dims=[1,2])
            U_data.append(U_datum_270)
            U_target.append([0,0,0,1]) # 270，标签为3
            count_U += 4
        if count_U + self.loader_L.batch_size > self.batch_size:
            U_data = U_data[:self.batch_size-self.loader_L.batch_size]
            U_target = U_target[:self.batch_size-self.loader_L.batch_size]
        
        U_data = torch.stack(U_data,dim=0)
        U_target = torch.tensor(U_target,dtype=torch.float)
        return L_data, L_target, U_data, U_target
    
    def __len__(self): # 计算该Loader的长度，为L和U中batch数量多的那个。
        L_BatchNum = len(self.loader_L)
        U_BatchNum = int(len(self.loader_U)*4 / (self.batch_size - self.loader_L.batch_size))
        return L_BatchNum if L_BatchNum > U_BatchNum else U_BatchNum
    
    def __iter__(self):
        self.loader_L_it = iter(self.loader_L)
        self.loader_U_it = iter(self.loader_U)
        return self

if __name__ == "__main__":
    from CIFAR10.dataset import CIFAR10DatasetAL
    dataset_L, dataset_U = CIFAR10DatasetAL.createLURandomly(nL=5000, image_size=32)
    loader = ComposeLoaderLU(dataset_L, dataset_U, 64)
    ite_loader = iter(loader)
    
    for i in range(len(loader)):
        L_data, L_target, U_data, U_target = next(ite_loader)
        # print(L_data.shape)
        # print(L_target.shape)
        # print(U_data.shape)
        # print(U_target.shape)
        print("\r{}".format(i), end="")
        
    # L_data, L_target, U_data, U_target = next(loader)
    # print(L_data.shape)
    # print(L_target.shape)
    # print(U_data.shape)
    # print(U_target.shape)
    
    # U_data = U_data.numpy()
    # U_data = np.transpose(U_data, (0,2,3,1))
    # U_data = (U_data * 255).astype(np.uint8)
    
    # for i in range(4):
    #     img_np = U_data[i,...]
    #     img = Image.fromarray(img_np)
    #     img.save("/home/wyx/vscode_projects/SSAL/test/{:02d}.jpg".format(i))
    # print(U_target[:4])
    
    
    
    