import os
import logging
import torch
from rrdbnet import RRDBNet as net
import cv2
import numpy as np
import argparse
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--folder-path', type=str, required=True, help='Path of the folder containing images.')
args = parser.parse_args()

# Helper function to read images
def imread_uint(path, n_channels=3):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2 and n_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    elif img.ndim == 3 and n_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

# Convert uint image to tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

# Convert tensor back to uint image
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Load model
model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load("Weights/G_GAN.pth"), strict=True)
model.eval()

# Disable gradient calculation
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

torch.cuda.empty_cache()

# Ensure output folder exists
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Process images in the folder
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Allowed image extensions
for image_name in os.listdir(args.folder-path):
    if image_name.lower().endswith(image_extensions):
        image_path = os.path.join(args.folder-path, image_name)
        logging.info(f'Processing image: {image_name}')

        # Read image
        image = imread_uint(image_path)

        # Convert image to tensor and send to GPU
        img_L = uint2tensor4(image)
        img_L = img_L.to(device)

        # Perform model inference
        img_E = model(img_L)

        # Convert tensor back to image
        img_E = tensor2uint(img_E)

        # Save output image
        output_name = os.path.join(output_folder, f'out_{image_name}')
        cv2.imwrite(output_name, img_E)

        logging.info(f'Output saved: {output_name}')

logging.info('\nAll images processed. Outputs are in the "outputs" folder.')
