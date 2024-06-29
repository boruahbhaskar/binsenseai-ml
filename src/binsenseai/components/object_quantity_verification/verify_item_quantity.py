import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import resnet50


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
if os.path.isfile('./src/binsenseai/components/object_quantity_verification/snapshots/resnet50_best.pth.tar'):
    model_path = "./src/binsenseai/components/object_quantity_verification/snapshots/resnet50_best.pth.tar"


    checkpoint = torch.load(model_path, map_location=torch.device(device))
    #print(checkpoint)
    model = checkpoint['model']
    print('model loaded')
    model.to(device)
    model.eval()  # Set the model to evaluation mode

# Define image preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
def verify_item_quantity(image,asin,input_quantity):
    # Load and preprocess the image
    image_dir = './artifacts/data_ingestion/amazon_bin_images'
    #'/home/ubuntu/abid/dataset/bin-images-resize'
    image_path = os.path.join(image_dir, image)
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
 
    # Get the ground truth quantity from the metadata
    # use the image_id to get metadata file to extract actual quantity for the item 
    # for the time being , consider input_quantity

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
# Forward pass through the model
    # with torch.no_grad():
    #     output = model(image)

    # Make predictions using the trained model

    predicted_quantity = torch.round(output)

    # Decide whether the object is present or not based on probability
    # Verify the item quantity
    _, predicted =  torch.max(output.detach(),1)
    #print(predicted_quantity, input_quantity,predicted )
    if predicted == input_quantity:
        return True
    else:
        return False
