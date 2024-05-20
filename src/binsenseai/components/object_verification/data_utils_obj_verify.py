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

class ImageFolderSiamesePair(data.Dataset):
    def __init__(self, root, train_file, pos_prob, transform=None, target_transform=None,
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
        self.pos_prob = pos_prob
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
        obj_list = train_item[1]
        #img1_path = '%05d.jpg' % (train_item[0]+1)
        
        img1_path=self.metadata_list[train_item[0]]['IMAGE_NAME']
        
        

        # pick up the pairing image, 
        # if positive it should have at least one commone object
        # if negative it should not have common object
        target = self.rand.binomial(1,self.pos_prob)
        if target == 1:
          obj_id = self.rand.randint(0,len(obj_list))
          target_list = obj_list[obj_id][1]
          img2_idx = target_list[self.rand.randint(0,len(target_list))]+1
          #img2_path = '%05d.jpg' % img2_idx
          img2_path=self.metadata_list[img2_idx]['IMAGE_NAME']
        else:
          temp = self.rand.randint(0,self.N_train)
          img2_idx = self.train_list[temp][0]+1
          #img2_path = '%05d.jpg' % (img2_idx)
          img2_path=self.metadata_list[img2_idx]['IMAGE_NAME']
          
        full_img_path1 = os.path.join(self.img_dir, img1_path)
        #print(full_img_path)
        img1 = self.loader(full_img_path1)  

        #img1 = self.loader(os.path.join(self.root, img1_path))
        
        full_img_path2 = os.path.join(self.img_dir, img2_path)
        img2 = self.loader(full_img_path2)  
        #img2 = self.loader(os.path.join(self.root, img2_path))
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return img1, img2, target
    
    def __len__(self):
        return self.N_train

class ImageFolderSiamesePairVal(data.Dataset):
    def __init__(self, root, val_file, max_target, transform=None, target_transform=None,
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
        
        img_path1=self.metadata_list[item[0]]['IMAGE_NAME']
        
        full_img_path1 = os.path.join(self.img_dir, img_path1)
        img1 = self.loader(full_img_path1)
        
        if self.transform is not None:
            img1 = self.transform(img1)

        target_list = item[3]
        n_target = len(target_list)
        if n_target > self.max_target:
          n_target = self.max_target
        target_batch = torch.Tensor(n_target)
        img1_batch = torch.Tensor(n_target,img1.size(0),img1.size(1),img1.size(2))
        img2_batch = torch.Tensor(n_target,img1.size(0),img1.size(1),img1.size(2))
        
        # pick up the target image in the training set
        for i in range(n_target):
          
          img2_idx = target_list[self.rand.randint(0,len(target_list))]+1
          
          if img2_idx == self.len_metadata :
            print(" img2_idx reached metadata last index, assign a random index")
            img2_idx = self.rand.randint(0,self.len_metadata)
          
          img_path2=self.metadata_list[img2_idx]['IMAGE_NAME']
          
          img2 = self.loader(os.path.join(self.img_dir, img_path2))
          
          if self.transform is not None:
            img2 = self.transform(img2)
            
          img1_batch[i].copy_(img1)
          img2_batch[i].copy_(img2)
          target_batch[i] = item[2]
        
        if self.target_transform is not None:
            target = self.target_transform(target_batch)
    
        return img1_batch, img2_batch, target_batch
    
    def __len__(self):
        return self.N_val