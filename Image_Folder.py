import os
import shutil
import logging
import torch
from rrdbnet import RRDBNet as net
import cv2
import numpy as np
from google.colab import files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the target directory for uploaded files
target_path = '/content/G_ESRGAN/inputs'
os.makedirs(target_path, exist_ok=True)

# Upload files through Colab's file upload interface
uploaded_files = files.upload()

# Move the uploaded files to the target directory
for file_name in uploaded_files.keys():
    source_path = file_name  # Uploaded file is in the current working directory
    dest_path = os.path.join(target_path, file_name)  # Destination path for the file
    shutil.move(source_path, dest_path)  # Move the file to the target directory

print(f"All images have been moved to {target_path}")

# Define image processing functions
def imread_uint(path, n_channels=3):
    if n_channels == 1:
        img = np.expand_dims(img, axis=2)  
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

# Set up the device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load("Weights/G_GAN.pth"), strict=True)
model.eval()

for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
torch.cuda.empty_cache()

# Ensure output folder exists
output_folder = "/content/G_ESRGAN/outputs"
os.makedirs(output_folder, exist_ok=True)

# Process each uploaded image in the target folder
for filename in os.listdir(target_path):
    image_path = os.path.join(target_path, filename)
    
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        
        if image is None:
            logging.warning(f"Skipping {filename}, not a valid image.")
            continue
        
        img_L = uint2tensor4(image)
        img_L = img_L.to(device)

        img_E = model(img_L)
        img_E = tensor2uint(img_E)

        output_name = os.path.join(output_folder, f"out_{filename}")
        cv2.imwrite(output_name, img_E)
        logging.info(f"Processed and saved: {output_name}")

logging.info('All images processed. Outputs saved in the outputs folder.')
