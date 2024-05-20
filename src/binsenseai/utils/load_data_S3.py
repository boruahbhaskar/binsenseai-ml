import os
from os import listdir
import os.path

import urllib.request as request
import zipfile
from src.binsenseai import logger
from src.binsenseai.utils.common import get_size
from pathlib import Path
from src.binsenseai.entity.config_entity import (DataIngestionConfig)
import boto3
import pandas as pd
import numpy as np
import json
import random

class DataReadAndTransformation:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.hard = False
        # Define the source and destination bucket names
        self.src_bucket = 'aft-vbi-pds'
        self.dst_bucket = 'binsenseai-s3'

        self.image_folder_prefix = 'bin-images'
        self.metadata_folder_prefix = 'metadata'

        # Example usage
        self.file_path = "artifacts\data_ingestion\binsimages.csv"  # Path  binsimages.csv s3://binsenseai-s3/binsimages.zip
        self.sheet_name = "Sheet1"        # Name of the sheet containing  data "Sheet1"  
    
    def read_excel_to_list(self,file_path, sheet_name):
        print(file_path, sheet_name)
        try:
            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_path, sheet_name=sheet_name,dtype={'image_name': str})
            
            # Convert DataFrame to a list of dictionaries
            data_list = df.to_dict(orient='records')
            print(data_list,len(data_list))
            return data_list
    
        except Exception as e:
            print("An error occurred:", str(e))
            return None
        
    def convert_to_string_with_zeros(self,number):
        print(number)
        # Determine the width of the number
        width = len(str(number))
        
        # Convert the number to a string and preserve leading zeros using string formatting
        string_with_zeros = "{:0{width}d}".format(number, width=width)

        return string_with_zeros
    
    def s3_read_write(self,data_list):
        
        # Set AWS access keys
        access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Initialize S3 client
        s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,region_name='us-east-1')

        # Specify the S3 bucket name and folder (prefix)
        bucket_name = 'aft-vbi-pds'
        image_folder_prefix = 'bin-images'
        metadata_folder_prefix = 'metadata'

        if data_list:
            print("Data successfully read from Excel file: " , len(data_list))
            
            counter = 0 
            for row in data_list:
                #fileName = convert_to_string_with_zeros(row['image_name']) # str(row['image_name'])
                fileName= str(row['image_name']) 
                str_jpeg ="jpg"
                str_json ="json"
            
                image_name  = ".".join([fileName, str_jpeg])
                metadata_file = ".".join([fileName, str_json])
                
                full_image_name = "/".join([image_folder_prefix , image_name])
                full_metdata_name = "/".join([metadata_folder_prefix , metadata_file])
                
                local_path_image = os.path.join('artifacts/data_ingestion/amazon_bin_images', image_name)
                
                local_path_metadata = os.path.join('artifacts/data_ingestion/amazon_bin_metadata', metadata_file)
            
                try:
                    s3.download_file(bucket_name, full_image_name, local_path_image)
                except Exception as e:
                    print(f"Error downloading {full_image_name}: {e}")
                
                try:
                    s3.download_file(bucket_name, full_metdata_name, local_path_metadata)
                except Exception as e:
                    print(f"Error downloading {full_metdata_name}: {e}")
                    
                counter += 1    
                print(f"Downloaded - {counter} images ")    
            else:
                print("Failed to read data from Excel file.")
        return True
    
        # getting whole metadata list 
    
    def get_metadata(self,img_dir,meta_dir):
        metadata=[]
        n_images=0
        metadata_list = listdir(meta_dir)
        N = len(metadata_list)
        # print(metadata_list[1])
        # N = 3876
        for i in range(N): 
            if i%500 == 0:
                print("get_metadata: processing (%d/%d)..." % (i,N))
        
            meta_file = metadata_list[i]
            json_path = os.path.join(meta_dir, meta_file)
            img = meta_file.replace("json","jpg")
            jpg_path = os.path.join(img_dir, img)
            

            if os.path.isfile(jpg_path) and os.path.isfile(json_path):
                d = json.loads(open(json_path).read())
                d["IMAGE_NAME"] = img
                metadata.append(d)
                n_images = n_images + 1
            else:
                metadata.append({})
                
                
        print("get_metadata: Available Images: %d" % n_images)
        return metadata,n_images

    # getting instance list
    def get_instance_data(self,metadata):
        instances={}
        N = len(metadata)
        for i in range(N):
            if i%1000 == 0:
                print("get_instance_data: processing (%d/%d)..." % (i,N))
            if metadata[i]:
                quantity = metadata[i]['EXPECTED_QUANTITY']
                bin_image = metadata[i]['IMAGE_NAME']
                if quantity>0:
                    bin_info = metadata[i]['BIN_FCSKU_DATA']
                    bin_keys = list(bin_info.keys())
                    for j in range(0,len(bin_info)):
                        instance_info = bin_info[bin_keys[j]]
                        asin = instance_info['asin']
                        if asin in instances:
                            # occurance
                            instances[asin]['repeat'] = instances[asin]['repeat'] + 1
                            # quantity
                            instances[asin]['quantity'] = instances[asin]['quantity'] + instance_info['quantity']
                            instances[asin]['bin_image_list'].append(bin_image)
                            instances[asin]['bin_list'].append(i)
                        else:
                            instances[asin]={}
                            instances[asin]['repeat'] = 1
                            instances[asin]['quantity'] = instance_info['quantity']
                            instances[asin]['name'] = instance_info['name']
                            bin_img_list = []
                            bin_img_list.append(bin_image)
                            instances[asin]['bin_image_list'] = bin_img_list
                            bin_list = []
                            bin_list.append(i)
                            instances[asin]['bin_list'] = bin_list
        return instances

    # make both  metadata and instances
    def make_metadata(self,img_dir, meta_dir):
        metadata,n_images = self.get_metadata(img_dir, meta_dir)
        instances = self.get_instance_data(metadata)
        # dumping out all metadata into a file
        print("dumping metadata.json...")
        with open('artifacts/data_transformation/metadata.json','w') as fp:
            json.dump(metadata,fp)
        print("dumping instances.json...")
        with open('artifacts/data_transformation/instances.json','w') as fp:
            json.dump(instances,fp)

    def split_train_val_data(self,img_dir, meta_dir):
        
        img_list = listdir(img_dir)
        N = len(img_list)
        
        json_list = listdir(meta_dir)
        print(json_list,N)
        list_random = list(range(N)) # bms error TODO remove list
        random.shuffle(list_random)

        # finding images that metadata exists
        meta_avail = np.zeros(N, dtype=bool)
        listIds= [0] * N#np.zeros(N, dtype=np.str)
        for i in range(N):
            json_path = json_list[i]
            json_path_joined = os.path.join(meta_dir,json_path)
            if os.path.isfile(json_path_joined):
                meta_avail[i] = True
                listIds[i] = json_path.split('.')[0]

        # assign validataion set
        valset = np.zeros(N, dtype=bool)
        n_valset = int(round(N*0.1))
        count = 0
        random.shuffle(list_random)
        for i in range(N):
            idx = list_random[i]
            if meta_avail[idx]:
                valset[idx]=True
                count = count + 1
                if count == n_valset:
                    break

        # writing out to textfile
        train_f = open('artifacts/data_transformation/random_train.txt','w')
        val_f = open('artifacts/data_transformation/random_val.txt','w')
        for i in range(N):
            if meta_avail[i]:
                if valset[i]:
                    val_f.write("\n"+listIds[i])
                else:
                    train_f.write("\n"+listIds[i])
        train_f.close()
        val_f.close()


    '''
        random_train = "random_train.txt"
        random_val = "random_val.txt"
        metadata_file = "metadata.json"
        instance_file = "instances.json"
    '''
    # get quantity given index
    def get_expected_quantity(self,idx,metadata):
        quantity = 0
        if metadata[idx]:
            quantity = metadata[idx]['EXPECTED_QUANTITY']
        return quantity
    
    def get_candidates(self,split_file,metadata):
        print("loading random split file")
        linenum = 0
        candidates = np.zeros(len(metadata),bool)
        with open(split_file) as f:
            for line in f.readlines():
                idx = int(line)-1 # check the content of the line
                quantity = self.get_expected_quantity(linenum,metadata)
                if quantity > 0:
                        candidates[linenum] = True
                linenum = linenum +1
        return candidates
    
    def generate_random_number(self,exclude_number):
        number = random.randint(1, 10)
        while number == exclude_number:
            number = random.randint(1, 10)
        return number
    
    def get_train_list(self,train_candidates,metadata,instances):
        train_list = []
        for idx in range(len(metadata)):
            if train_candidates[idx]:
                bin_info = metadata[idx]['BIN_FCSKU_DATA']
                bin_keys = list(bin_info.keys())
                object_list = []
                # iterate over objects in the bin
                for j in range(0,len(bin_info)):
                    asin = bin_info[bin_keys[j]]['asin']
                    
                    
                    if asin==None:
                        continue
                    actual_quantity = bin_info[bin_keys[j]]['quantity']
                    repeat_in = 0
                    target_idx_list = []
                    # how many this object repeted in the candidate images?
                    for target_idx in instances[asin]['bin_list']:
                        if train_candidates[target_idx]==True and target_idx!=idx:
                            repeat_in = repeat_in + 1
                            target_idx_list.append(target_idx)
                    # if repeat > 1, then at least twice showed up, we can make a
                    # positive pair with this object
                    # if repeat_in > 1:
                    #     object_list.append([target_idx_list])
                    train_list.append([idx,asin, actual_quantity,target_idx_list])
        return train_list

    def get_val_pos_list(self,train_candidates, val_candidates,metadata,instances):
   
        val_list = []
        for idx in range(len(metadata)):
            if val_candidates[idx]:
                bin_info = metadata[idx]['BIN_FCSKU_DATA']
                bin_keys = list(bin_info.keys())
                
                # iterate over objects in the bin
                for j in range(0,len(bin_info)):
                    
                    asin = bin_info[bin_keys[j]]['asin']
                    actual_quantity = bin_info[bin_keys[j]]['quantity']
                    if asin==None:
                        continue
                    target_idx_list = []
                    for target_idx in instances[asin]['bin_list']:
                        if train_candidates[target_idx]==True and target_idx!=idx:
                            target_idx_list.append(target_idx)
                    if target_idx_list:
                        val_list.append([idx, asin, 1, actual_quantity, actual_quantity , target_idx_list])
                        
        return val_list
    
    def get_val_neg_list(self,train_candidates, val_candidates,metadata,instances):
        val_list = []
    
        
        for idx in range(len(metadata)):
            if val_candidates[idx]:
                # pick up random object asin from the list
                bin_info = metadata[idx]['BIN_FCSKU_DATA']
                bin_keys = list(bin_info.keys())
                
                # iterate over objects in the bin
                for j in range(0,len(bin_info)):
                    
                    asin = bin_info[bin_keys[j]]['asin']
                    actual_quantity = bin_info[bin_keys[j]]['quantity']
                    if asin==None:
                        continue
                    
                    target_idx_list = []
                    
                    for target_idx in instances[asin]['bin_list']:
                        
                        if train_candidates[target_idx]==True and target_idx!=idx:
                            target_idx_list.append(target_idx)
                            
                    if target_idx_list:
                        val_list.append([idx, asin, 0, actual_quantity, self.generate_random_number(actual_quantity) , target_idx_list])
                        
                    
        return val_list

    def make_obj_num_verification_data(self,random_train,random_val,metadata_file,instance_file):
        print("loading metadata!")
        with open(metadata_file) as json_file:
            metadata = json.load(json_file)
            
        print("loading instance data!")
        with open(instance_file) as json_file:
            instances = json.load(json_file)
            
        instance_keys = instances.keys()
        N = len(metadata)
        N_inst = len(instance_keys)

        train_candidates = self.get_candidates(random_train,metadata)
        val_candidates = self.get_candidates(random_val,metadata)

        # building training sets
        train_list = self.get_train_list(train_candidates,metadata,instances)

        # building validataion sets
        val_pos_list = self.get_val_pos_list(train_candidates, val_candidates,metadata,instances)
        val_neg_list = self.get_val_neg_list(train_candidates, val_candidates,metadata,instances)
        
        pos_samples = random.sample(val_pos_list,len(val_pos_list))
        neg_samples = random.sample(val_neg_list,len(val_neg_list))
        
        val_list = pos_samples + neg_samples
        random.shuffle(val_list)
        
        print("dumping train and val sets into json file")
        
        out_fname = 'artifacts/data_transformation/obj_num_verification_train.json'
        with open(out_fname,'w') as f: ## wb to w
            json.dump(train_list,f)

        out_fname = 'artifacts/data_transformation/obj_num_verification_val.json'
        with open(out_fname,'w') as f: ## wb to w
            json.dump(val_list,f)    

    
    # ------------- #

    # get quantity given index
    def get_quantity(self,idx,metadata):
        quantity = 0
        if metadata[idx]:
            quantity = metadata[idx]['EXPECTED_QUANTITY']
        return quantity

    # True or False array
    def get_candidates(self,split_file,metadata):
        print("loading random split file")
        linenum = 0
        candidates = np.zeros(len(metadata),bool)
        with open(split_file) as f:
            for line in f.readlines():
                idx = int(line)-1
                quantity = self.get_quantity(linenum,metadata)
                if quantity > 0 and quantity < 6:
                        candidates[linenum] = True
                linenum = linenum +1
        print(candidates,split_file)
        return candidates

    # making split train list
    # each element, it contains (image idx, object list)
    # for each objects, it contains list of indices of images that contain the object
    def get_train_list(self,train_candidates,metadata,instances):
        train_list = []
        linenum = 0
        for idx in range(len(metadata)):
            if train_candidates[idx]:
                bin_info = metadata[idx]['BIN_FCSKU_DATA']
                bin_keys = list(bin_info.keys())
                object_list = []
                # iterate over objects in the bin
                for j in range(0,len(bin_info)):
                    asin = bin_info[bin_keys[j]]['asin']
                    if asin==None:
                        continue
                    repeat_in = 0
                    target_idx_list = []
                    # how many this object repeted in the candidate images?
                    for target_idx in instances[asin]['bin_list']:
                        if train_candidates[target_idx]==True and target_idx!=idx:
                            repeat_in = repeat_in + 1
                            target_idx_list.append(target_idx)
                    # if repeat > 1, then at least twice showed up, we can make a
                    # positive pair with this object
                    if repeat_in > 1:
                        object_list.append([asin,target_idx_list])
                if object_list: 
                    train_list.append([idx,object_list])
        return train_list

    # making split val list(for positive pair)
    # each element, it contains (image idx, instance asin, pos or neg, target index list),
    def get_val_pos_list(self,train_candidates, val_candidates,metadata,instances):
        val_list = []
        for idx in range(len(metadata)):
            if val_candidates[idx]:
                bin_info = metadata[idx]['BIN_FCSKU_DATA']
                bin_keys = list(bin_info.keys())
                # iterate over objects in the bin
                for j in range(0,len(bin_info)):
                    asin = bin_info[bin_keys[j]]['asin']
                    if asin==None:
                        continue
                    target_idx_list = []
                    for target_idx in instances[asin]['bin_list']:
                        if train_candidates[target_idx]==True and target_idx!=idx:
                            target_idx_list.append(target_idx)
                    if target_idx_list:
                        val_list.append([idx, asin, 1, target_idx_list])
        return val_list 

    # making split val list(for negative pair)
    def get_val_neg_list(self,train_candidates, val_candidates,metadata,instances):
        instance_keys = instances.keys()
        val_list = []
        for idx in list(range(len(metadata))):
            if val_candidates[idx]:
                # pick up random object asin
                asin = list(instance_keys)[random.randint(0,len(instances)-1)]
                if asin==None:
                    continue
                inst = instances[asin]
                target_idx_list = []
                for target_idx in inst['bin_list']:
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        target_idx_list.append(target_idx)
                if len(target_idx_list) > 0:
                    val_list.append([idx, asin, 0, target_idx_list])
        return val_list 



    def make_obj_verification_data(self,random_train,random_val,metadata_file,instance_file):
        print("loading metadata!")
        with open(metadata_file) as json_file:
            metadata = json.load(json_file)
        print("loading instance data!")
        with open(instance_file) as json_file:
            instances = json.load(json_file)
        instance_keys = instances.keys()
        N = len(metadata)
        N_inst = len(instance_keys)
        print(N_inst,N)
        train_candidates = self.get_candidates(random_train,metadata)
        val_candidates = self.get_candidates(random_val,metadata)

        # building training sets
        train_list = self.get_train_list(train_candidates,metadata,instances)

        # building validataion sets
        val_pos_list = self.get_val_pos_list(train_candidates, val_candidates,metadata,instances)
        val_neg_list = self.get_val_neg_list(train_candidates, val_candidates,metadata,instances)
        pos_samples = random.sample(val_pos_list,200)
        neg_samples = random.sample(val_neg_list,200)
        val_list = pos_samples + neg_samples
        random.shuffle(val_list)
        print("dumping train and val sets into json file")
        if self.hard:
            out_fname = 'artifacts/data_transformation/obj_verification_train_hard.json'
        else:
            out_fname = 'artifacts/data_transformation/obj_verification_train.json'
        with open(out_fname,'wt') as f:
            json.dump(train_list,f)

        if self.hard:
            out_fname = 'artifacts/data_transformation/obj_verification_val_hard.json'
        else:
            out_fname = 'artifacts/data_transformation/obj_verification_val.json'
        with open(out_fname,'wt') as f:
            json.dump(val_list,f)