#-*- encoding:utf-8
import os, sys, json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms,datasets
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class TinyImageNetDataset(Dataset):
    
    C = 200
    # HLevel: it can be 1,2,3 or [1,2,3]
    # HLevel means the obtained level of labels
    def __init__(self, train=True, image_size=64) -> None:
        super().__init__()
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transform = transform_train if train else transform_test
        self.train = train
        self.load_images()
    
    def load_images(self):
        #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
        # deal with training set
        root_path = "/home/wyx/datasets/TinyImageNet"
        Y_train_t = []
        train_img_names = []
        train_imgs = []
        
        with open(os.path.join(root_path, 'tiny-imagenet-200/wnids.txt')) as wnid:
            for line in wnid:
                Y_train_t.append(line.strip('\n'))
        for Y in Y_train_t:
            Y_path = os.path.join(root_path, 'tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt')
            train_img_name = []
            with open(Y_path) as Y_p:
                for line in Y_p:
                    train_img_name.append(line.strip('\n').split('\t')[0])
            train_img_names.append(train_img_name)
        train_labels = np.arange(200)
        idx = 0
        for Y in Y_train_t:
            train_img = []
            for img_name in train_img_names[idx]:
                img_path = os.path.join(root_path, 'tiny-imagenet-200/train/', Y, 'images', img_name)
                # cv读取和PIL读取只有BGR和RGB的区别，对模型性能没有影响
                # 我们仍然将其转化为RGB格式
                im_bgr = cv2.imread(img_path)
                im_rgb = cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
                train_img.append(im_rgb)
            train_imgs.append(train_img)
            idx = idx + 1
        train_imgs = np.array(train_imgs)
        train_imgs = train_imgs.reshape(-1, 64, 64, 3)
        X_tr = []
        Y_tr = []
        for i in range(train_imgs.shape[0]):
            Y_tr.append(i//500)
            X_tr.append(train_imgs[i])
        #X_tr = torch.from_numpy(np.array(X_tr))
        X_tr = np.array(X_tr)
        #Y_tr = torch.from_numpy(np.array(Y_tr)).long()
        Y_tr = np.asarray(Y_tr,dtype=np.int32)

        #deal with testing (val) set
        Y_test_t = []
        Y_test = []
        test_img_names = []
        test_imgs = []
        with open(os.path.join(root_path, 'tiny-imagenet-200/val/val_annotations.txt')) as val:
            for line in val:
                test_img_names.append(line.strip('\n').split('\t')[0])
                Y_test_t.append(line.strip('\n').split('\t')[1])
        for i in range(len(Y_test_t)):
            for i_t in range(len(Y_train_t)):
                if Y_test_t[i] == Y_train_t[i_t]:
                    Y_test.append(i_t)
        test_labels = np.array(Y_test)
        test_imgs = []
        for img_name in test_img_names:
            img_path = os.path.join(root_path, 'tiny-imagenet-200/val/images', img_name)
            im_bgr = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
            test_imgs.append(im_rgb)
            
        test_imgs = np.array(test_imgs)
        X_te = []
        Y_te = []

        for i in range(test_imgs.shape[0]):
            X_te.append(test_imgs[i])
            Y_te.append(Y_test[i])
        #X_te = torch.from_numpy(np.array(X_te))
        X_te = np.array(X_te)
        #Y_te = torch.from_numpy(np.array(Y_te)).long()
        Y_te = np.asarray(Y_te, dtype=np.int32)
        
        if self.train:
            self.images = X_tr
            self.labels = Y_tr    
        else:
            self.images = X_te
            self.labels = Y_te
        
    def __getitem__(self, index):
        image = self.images[index % len(self.labels)]
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).float()
        target = np.zeros(TinyImageNetDataset.C, dtype=np.float32)
        target[int(self.labels[index])] = 1
        return image, target
        
    def __len__(self):
        return len(self.labels)

class TinyImageNetDatasetAL(Dataset):
    
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
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
    def createLURandomly(nL=1000, image_size=64):
        #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
        # deal with training set
        root_path = "/home/wyx/datasets/TinyImageNet"
        Y_train_t = []
        train_img_names = []
        train_imgs = []
        
        with open(os.path.join(root_path, 'tiny-imagenet-200/wnids.txt')) as wnid:
            for line in wnid:
                Y_train_t.append(line.strip('\n'))
        for Y in Y_train_t:
            Y_path = os.path.join(root_path, 'tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt')
            train_img_name = []
            with open(Y_path) as Y_p:
                for line in Y_p:
                    train_img_name.append(line.strip('\n').split('\t')[0])
            train_img_names.append(train_img_name)
        train_labels = np.arange(200)
        idx = 0
        for Y in Y_train_t:
            train_img = []
            for img_name in train_img_names[idx]:
                img_path = os.path.join(root_path, 'tiny-imagenet-200/train/', Y, 'images', img_name)
                # cv读取和PIL读取只有BGR和RGB的区别，对模型性能没有影响
                # 我们仍然将其转化为RGB格式
                im_bgr = cv2.imread(img_path)
                im_rgb = cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
                train_img.append(im_rgb)
            train_imgs.append(train_img)
            idx = idx + 1
        train_imgs = np.asarray(train_imgs)
        train_imgs = train_imgs.reshape(-1, 64, 64, 3)
        X_tr = []
        Y_tr = []
        for i in range(train_imgs.shape[0]):
            Y_tr.append(i//500)
            X_tr.append(train_imgs[i])
        #X_tr = torch.from_numpy(np.array(X_tr))
        X_tr = np.asarray(X_tr, dtype=np.uint8)
        #Y_tr = torch.from_numpy(np.array(Y_tr)).long()
        Y_tr = np.asarray(Y_tr,dtype=np.int32)
        
        data = [[] for _ in range(TinyImageNetDataset.C)] # Rerange by classes
        for i in range(X_tr.shape[0]):
            label = int(Y_tr[i])
            image = X_tr[i]
            data[label].append((label, image))

        C = TinyImageNetDataset.C
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
        L = TinyImageNetDatasetAL(L_data, "L", image_size)
        U = TinyImageNetDatasetAL(U_data, "U", image_size)
        return L, U

    def __getitem__(self, index):
        datum_id = self.dataidx[index]
        label,img = self.data[datum_id]
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img).float()
        target = np.zeros(TinyImageNetDataset.C, dtype=np.float32)
        target[label] = 1
        return img, target
        
    def __len__(self):
        return len(self.dataidx)

if __name__ == "__main__":
    data = TinyImageNetDataset(train=True,image_size=64)
    print(data.images.shape)
    print(data.labels.shape)