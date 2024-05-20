import json
import ast
import os
# def intersection(*lists):
#     return list(set.intersection(*map(set, lists)))

def intersection(lists):
    if not lists:
        return []
    intersection_set = set(lists[0])
    for lst in lists[1:]:
        intersection_set &= set(lst)
    return list(intersection_set)


instance_file = "./artifacts/data_transformation/instances.json"

def get_images(input_list):
    
    with open(instance_file) as json_file:
        instances = json.load(json_file)
     
    #print(input_list)  
    data = input_list 
    images_list =[]        
    for item in data:
        asin = item[0]
        #print(asin)
        images_list.append(instances[asin]['bin_image_list'])
             
    relevant_image_list = intersection(images_list)
    if len(relevant_image_list) > 0 :
        image_name =  relevant_image_list[0]
        image_dir = './artifacts/data_ingestion/amazon_bin_images'
        #'/home/ubuntu/abid/dataset/bin-images-resize'
        image_path = os.path.join(image_dir, image_name)
        return {"image_path": image_path, "image_name": image_name}
        
    
    return {"image_path": None, "image_name": None}#relevant_image_list
