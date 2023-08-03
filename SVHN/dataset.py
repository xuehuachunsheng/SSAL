#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms,datasets
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SVHNDataset(Dataset):
    
    C = 10
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    def __init__(self, train=True, image_size=32) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if train else transform_test
        self.train = train
        self.load_images()
    
    def load_images(self):
        data_train = datasets.SVHN('/home/wyx/datasets/SVHN', split='train', download=False)
        data_test = datasets.SVHN('/home/wyx/datasets/SVHN', split='test', download=False)
        data = data_train if self.train else data_test
        self.images = data.data
        self.labels = data.labels
    
    def __getitem__(self, index):
        image = self.images[index % len(self.labels)]
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).float()
        target = np.zeros(10, dtype=np.float32)
        target[int(self.labels[index])] = 1
        return image, target
        
    def __len__(self):
        return len(self.labels)

class SVHNDatasetAL(Dataset):
    
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    # data_type can be one in ["L", "U"]
    # L: initial labeled dataset
    # U: unlabled pool
    def __init__(self, data, data_type="L", image_size=32) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            #transforms.RandomResizedCrop(size=(image_size, image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.05),
            #transforms.RandomRotation(30),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if data_type=="L" else transform_test
        self.data = data
        self.dataidx = list(self.data) # Obtains all the datum ids
        self.data_type = data_type
    
    # Initial Labled Dataset is class balanced
    def createLURandomly(nL=500, image_size=32):
        data_train = datasets.SVHN('/home/wyx/datasets/SVHN', split='train', download=False)
        
        data = [[] for _ in range(10)] # Rerange by classes
        for i in range(data_train.data.shape[0]):
            label = int(data_train.labels[i])
            image = data_train.data[i]
            image = np.transpose(image, (1,2,0)).astype(np.uint8)
            data[label].append((label, image))

        C = SVHNDataset.C
        np.random.seed(0)
        for i in range(C):
            np.random.shuffle(data[i])
        c = nL / C # The number of samples of each class
        L_data = {}
        U_data = {}
        id = 0
        for i in range(C):
            for j in range(len(data[i])):
                if j < c:
                    L_data[str(id)] = data[i][j]
                else:
                    U_data[str(id)] = data[i][j]
                id += 1
        L = SVHNDatasetAL(L_data, "L", image_size)
        U = SVHNDatasetAL(U_data, "U", image_size)
        return L, U

    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        target = np.zeros(10, dtype=np.float32)
        target[label] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

if __name__ == "__main__":
    data_train = datasets.SVHN('/home/wyx/datasets/SVHN', split='train', download=True)
    print(data_train.data.shape)
    print(data_train.labels.shape)