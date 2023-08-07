#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FashionMNISTDataset(Dataset):
    
    # Note: classes_finest id corresponds to the origin data id
    # This sequence order cannot change!
    classes_finest = ["TShirt", "Trouser", "Pullover", "Dress", 
                        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]
    C = 10
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    def __init__(self, train=True, image_size=32) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.05),
            #transforms.RandomRotation(30),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if train else transform_test
        self.train = train
        self.load_images()
    
    def load_images(self):
        train_data_path = "/home/wyx/datasets/FashionMNIST/fashion-mnist_train.csv" 
        test_data_path = "/home/wyx/datasets/FashionMNIST/fashion-mnist_test.csv" 
        data_path = train_data_path if self.train else test_data_path
        f = open(data_path, "r")
        raw_data = f.readlines()
        n = len(raw_data) - 1
        f.close()
        images, labels = np.empty(shape=(n, 28, 28),dtype=np.uint8), np.empty(n, dtype=np.int32)
        for i, line in enumerate(raw_data[1:]):
            line = line.strip().split(",")
            labels[i] = int(line[0])
            images[i] = np.reshape([int(x) for x in line[1:]], newshape=(28,28))
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        target = np.zeros(10, dtype=np.float32)
        target[self.labels[index]] = 1
        return img, target
        
    def __len__(self):
        return len(self.labels)

class FashionMNISTDatasetAL(Dataset):
    
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
        data_path="/home/wyx/datasets/FashionMNIST/fashion-mnist_train.csv"
        f = open(data_path, "r")
        raw_data = f.readlines()
        f.close()
        data = [[] for _ in range(10)] # Rerange by classes
        for i, line in enumerate(raw_data[1:]):
            line = line.strip().split(",")
            label = int(line[0])
            image = np.reshape([int(x) for x in line[1:]], newshape=(28,28)).astype(np.uint8)
            data[label].append((label, image))

        C = len(FashionMNISTDataset.classes_finest)
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
        L = FashionMNISTDatasetAL(L_data, "L", image_size)
        U = FashionMNISTDatasetAL(U_data, "U", image_size)
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

def test_general():
    d = FashionMNISTDataset(train=True)
    img,label = d[8529]
    assert list(label) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

if __name__ == "__main__":
    test_general()
