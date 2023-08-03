#-*- encoding:utf-8
import os, sys, json
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIFAR100Dataset(Dataset):
    # 0,transport, animal, sky, water, road, bird, reptile, pet, medium
    classes_finest = ["Apples", "Aquarium fish", "Baby", "Bear", "Beaver", 
                      "Bed", "Bee", "Beetle", "Bicycle", "Bottles", 
                      "Bowls", "Boy", "Bridge", "Bus", "Butterfly", 
                      "Camel", "Cans", "Castle", "Caterpillar", "Cattle", 
                      "Chair", "Chimpanzee", "Clock", "Cloud", "Cockroach", 
                      "Couch", "Crab", "Crocodile", "Cups", "Dinosaur", 
                      "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", 
                      "Girl", "Hamster", "House", "Kangaroo", "Keyboard", 
                      "Lamp", "Lawn-mower", "Leopard", "Lion", "Lizard", 
                      "Lobster", "Man", "Maple", "Motorcycle", "Mountain", 
                      "Mouse", "Mushrooms", "Oak", "Oranges", "Orchids", 
                      "Otter", "Palm", "Pears", "Pickup truck", "Pine", 
                      "Plain", "Plates", "Poppies", "Porcupine", "Possum", 
                      "Rabbit", "Raccoon", "Ray", "Road", "Rocket", 
                      "Roses", "Sea", "Seal", "Shark", "Shrew", 
                      "Skunk", "Skyscraper", "Snail", "Snake", "Spider", 
                      "Squirrel", "Streetcar", "Sunflowers", "Sweet peppers", "Table", 
                      "Tank", "Telephone", "Television", "Tiger", "Tractor", 
                      "Train", "Trout", "Tulips", "Turtle", "Wardrobe", 
                      "Whale", "Willow", "Wolf", "Woman", "Worm"]
    C = 100
    def __init__(self, train=True, image_size=32):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(image_size,image_size)),
            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train = train
        self.image_size = image_size
        self.mat, self.labels = CIFAR100Dataset.load_data(train)
        self.transform = transform_train if train else transform_test

    def __getitem__(self, index):
        image = self.mat[index % len(self.labels)]
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).float()
        
        target = np.zeros(100, dtype=np.float32)
        target[int(self.labels[index])] = 1
        
        return image, target
    
    def __len__(self):
        return len(self.labels)
    
    def load_data(train=True):
        train_data_path = "/home/wyx/datasets/cifar-100-python/train"
        test_data_path = "/home/wyx/datasets/cifar-100-python/test"
        path = train_data_path if train else test_data_path
        f = open(path, "rb")
        dic = pickle.load(f, encoding="bytes")
        f.close()
        images = dic[b"data"]
        images = np.reshape(images, (-1, 3, 32, 32))
        finelabels = np.asarray(dic[b"fine_labels"])
        return images, finelabels

class CIFAR100DatasetAL(Dataset):
    
    # data_type can be one in ["L", "U"]
    # L: initial labeled dataset
    # U: unlabled pool
    def __init__(self, data, data_type="L", image_size=224) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(image_size,image_size)),
            transforms.RandomResizedCrop(size=(image_size,image_size),scale=(0.8,1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.transform = transform_train if data_type=="L" else transform_test
        self.data = data
        self.dataidx = list(self.data) # Obtains all the datum ids
        self.data_type = data_type
    
    # Initial Labled Dataset is class balanced
    def createLURandomly(nL=1000, image_size=224):
        path = "/home/wyx/datasets/cifar-100-python/train"
        f = open(path, "rb")
        dic = pickle.load(f, encoding="bytes")
        f.close()
        images = dic[b"data"]
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.transpose(images, (0,2,3,1)).astype(np.uint8)
        finelabels = np.asarray(dic[b"fine_labels"])
        
        C = CIFAR100Dataset.C
        data = [[] for _ in range(C)] # Rerange by classes
        for i in range(len(finelabels)):
            image, label = images[i], finelabels[i]
            data[int(label)].append((int(label), image))

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
        L = CIFAR100DatasetAL(L_data, "L", image_size)
        U = CIFAR100DatasetAL(U_data, "U", image_size)
        return L, U

    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        # one hot
        target = np.zeros(100, dtype=np.float32)
        target[label] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

class testIte:
    def __init__(self):
        self.x = 0
        self.y = 0
        
    def __next__(self):
        if self.x < 5:    
            x = self.x
            y = self.y * 2
            self.x += 1
            self.y += 1
            return x, y
        
    def __iter__(self):
        return self
