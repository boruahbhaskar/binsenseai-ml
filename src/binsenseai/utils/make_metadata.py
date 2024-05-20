import os
from os import listdir
import os.path
import numpy as np
import random
import json

img_dir= "/home/ubuntu/abid/dataset/amazon_bin_images/"
meta_dir = "/home/ubuntu/abid/dataset/amazon_bin_metadata/"

# getting whole metadata list 
def get_metadata(img_dir,meta_dir):
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
def get_instance_data(metadata):
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


if __name__ == '__main__':
    metadata,n_images = get_metadata(img_dir, meta_dir)
    instances = get_instance_data(metadata)
    # dumping out all metadata into a file
    print("dumping metadata.json...")
    with open('metadata.json','w') as fp:
        json.dump(metadata,fp)
    print("dumping instances.json...")
    with open('instances.json','w') as fp:
        json.dump(instances,fp)
