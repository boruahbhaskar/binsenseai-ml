import torch
from PIL import Image
from src.binsenseai.components.object_verification.siamese import siamese_resnet  # Assuming this is your model architecture
import torchvision.models as models
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

best_prec = 0
train_loss_list = []
val_acc_list = []

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model
print("=> Loading model '{}'".format('resnet34'))
cnn_model = models.__dict__['resnet34'](weights=None) #(pretrained=False)
net = siamese_resnet(cnn_model)
net.to(device)

#pickle_load_args = {'map_location': torch.device('cpu')}

if os.path.isfile('./src/binsenseai/components/object_verification/snapshots/resnet34_siamese_best.pth.tar'):
    checkpoint = torch.load('./src/binsenseai/components/object_verification/snapshots/resnet34_siamese_best.pth.tar',map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    best_prec = checkpoint['best_prec']
    train_loss_list = checkpoint['train_loss_list']
    val_acc_list = checkpoint['val_acc_list']
    params = net.parameters()

    # /Users/bboruah/Machine_Learning/projects/binsenseai/src/binsenseai/components/object_verification/snapshots/resnet34_siamese_best.pth.tar

    net.to(device)
    net.eval()  # Set the model to evaluation mode

# Define image preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def verify_object_presence(image_path1,image_path2):
    # Load and preprocess the image
    image_dir = './artifacts/data_ingestion/amazon_bin_images'
    #'/home/ubuntu/abid/dataset/bin-images-resize'
    image_path1 = os.path.join(image_dir, image_path1)
    image1 = Image.open(image_path1).convert("RGB")
    image1 = preprocess(image1).unsqueeze(0).to(device)  # Add batch dimension and move to device

    image_path2 = os.path.join(image_dir, image_path2)
    image2 = Image.open(image_path2).convert("RGB")
    image2 = preprocess(image2).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Extract object category from the question
    #object_category = question.split()[-2].lower()
    #print('object_category',object_category)
    # Forward pass through the model
    with torch.no_grad():
        output = net(image1, image2)
        #print(output)
    # Predicted probability of object presence
    probability = output.item()  # Assuming output is a single scalar representing probability
    probability = round(probability,2) 
    #answer = str(round(answer, 2))
    
    # Decide whether the object is present or not based on probability
    if probability >= 0.5:
        prediction = "yes"
    else:
        prediction = "no"
    
    return prediction