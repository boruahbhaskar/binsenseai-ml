import boto3
import os
import pandas as pd

# Set AWS access keys
aws_access_key = "AKIA4LQSYJICAVT3FQLQ"
aws_secret_key = "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"


## Download IK specific dataset 

def read_excel_to_list(file_path, sheet_name):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name,dtype={'image_name': str})
        
        # Convert DataFrame to a list of dictionaries
        data_list = df.to_dict(orient='records')
        
        return data_list
    
    except Exception as e:
        print("An error occurred:", str(e))
        return None

def convert_to_string_with_zeros(number):
    # Determine the width of the number
    width = len(str(number))
    
    # Convert the number to a string and preserve leading zeros using string formatting
    string_with_zeros = "{:0{width}d}".format(number, width=width)

    return string_with_zeros

# Example usage
file_path = "artifacts\data_ingestion\binsimages.csv"  # Path  binsimages.csv s3://binsenseai-s3/binsimages.zip
sheet_name = "Sheet1"        # Name of the sheet containing  data "Sheet1"  

data_list = read_excel_to_list(file_path, sheet_name)

print(data_list)

# Download the files from S3 

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

# Specify the S3 bucket name and folder (prefix)
bucket_name = 'aft-vbi-pds'
image_folder_prefix = 'bin-images'
metadata_folder_prefix = 'metadata'

#https://aft-vbi-pds.s3.amazonaws.com/bin-images/523.jpg 
#https://aft-vbi-pds.s3.amazonaws.com/metadata/523.json


# Create a local directory to save the downloaded image files
local_image_directory = "amazon_bin_images"
os.makedirs(local_image_directory, exist_ok=True)


#Create a local directory to save the downloaded metadata files
local_metadata_directory = "amazon_bin_metadata"
os.makedirs(local_metadata_directory, exist_ok=True)


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
        
        local_path_image = os.path.join(local_image_directory, image_name)
        
        local_path_metadata = os.path.join(local_metadata_directory, metadata_file)
        
        
       # print(local_path_image)
        
       # print(local_path_metadata)
       
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
        #print(f"Downloaded {full_image_name} to {local_path_image}")
        #print(f"Downloaded {full_metdata_name} to {local_path_metadata}")
else:
    print("Failed to read data from Excel file.")
    

print("All specified images in the 'bin-images' folder have been listed successfully.")

# artifacts for excel sheet 
# load read the excel shhet
# save it into S3 bucket

