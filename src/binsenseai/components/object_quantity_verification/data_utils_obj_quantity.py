import torch.utils.data as data
import torch

from PIL import Image
import numpy as np
import os
import os.path
import json
import random

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolderTraining(data.Dataset):
    def __init__(self, root, train_file, transform=None, target_transform=None,
                 loader=default_loader):
        
        print('loading train_list json file...')
        with open(train_file) as f:
          train_list = json.loads(f.read())
        print('finished loading train_list json file')
        
        metadata_file = "./artifacts/data_transformation/metadata.json"
        img_dir = "./artifacts/data_ingestion/bin-images-resize"

        print("loading metadata!")
        
        with open(metadata_file) as json_file:
            metadata_list = json.load(json_file)
        print('finished loading metadata json file')
        
        
        self.root = root
        self.train_list = train_list
        #self.pos_prob = pos_prob
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.N_train = len(self.train_list)
        self.rand = np.random.RandomState()
        self.metadata_list = metadata_list
        self.img_dir = img_dir
        
    
    def __getitem__(self, index):
        # pick up the training image
        train_item = self.train_list[index]
        #obj_list = train_item[1]
        #img1_path = '%05d.jpg' % (train_item[0]+1)
        target = int(train_item[2])
        img_path=self.metadata_list[train_item[0]]['IMAGE_NAME']
        
        full_img_path = os.path.join(self.img_dir, img_path)
        #print(full_img_path)
        img = self.loader(full_img_path)  

        if self.transform is not None:
            img = self.transform(img)
  
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return img, target
    
    def __len__(self):
        return self.N_train

class ImageFolderValidation(data.Dataset):
    def __init__(self, root, val_file,max_target ,transform=None, target_transform=None,
                 loader=default_loader):
        print('loading split list json file...')
        with open(val_file) as f:
          val_list = json.loads(f.read())
        print('finished loading val_list json file')
        
        metadata_file = "./artifacts/data_transformation/metadata.json"
        img_dir = "./artifacts/data_ingestion/bin-images-resize"   
        print("loading metadata!")
        with open(metadata_file) as json_file:
            metadata_list = json.load(json_file)
        print('finished loading metadata json file')
        
        len_metadata = len(metadata_list)
        
        self.root = root
        self.val_list = val_list
        self.max_target = max_target
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.N_val = len(self.val_list)
        self.rand = np.random.RandomState()
        self.metadata_list = metadata_list
        self.img_dir = img_dir
        self.len_metadata = len_metadata
    
    def __getitem__(self, index):
        # pick up the validation image
        item = self.val_list[index]
        
        target = int(item[3])## changed to input quantity
        
        img_path=self.metadata_list[item[0]]['IMAGE_NAME']

        full_img_path = os.path.join(self.img_dir, img_path)
        #print(full_img_path)
        img = self.loader(full_img_path)
        
        # if self.transform is not None:
        #     img1 = self.transform(img)

        # target_list = item[5]
        # n_target = len(target_list)
        
        # if n_target > self.max_target:
        #   n_target = self.max_target
          
        # target_batch = torch.Tensor(n_target)
        # img1_batch = torch.Tensor(n_target,img1.size(0),img1.size(1),img1.size(2))
   
        # # pick up the target image in the training set
        # for i in range(n_target):  
        #   img1_batch[i].copy_(img1)
        #   target_batch[i] = item[3]
        
        # if self.target_transform is not None:
        #     target_batch = self.target_transform(target_batch)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return self.N_val