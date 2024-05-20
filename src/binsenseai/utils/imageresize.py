from PIL import Image
from os import listdir
import os.path

img_dir = "../dataset/amazon_bin_images/"
resized_img_dir = "../dataset/bin-images-resize/"

if not os.path.exists(resized_img_dir):
    os.makedirs(resized_img_dir)

img_list = listdir(img_dir)
N = len(img_list)
#N = 535234
count = 0
for filename  in img_list:
    jpg_path = os.path.join(img_dir,filename)
    resized_jpg_path = os.path.join(resized_img_dir,filename)
    img = Image.open(jpg_path).convert('RGB')
    resized_img = img.resize((224,224),Image.BILINEAR)
    resized_img.save(resized_jpg_path)
    count = count + 1
    print('processed %s(%d/%d)' % (filename,count,N))