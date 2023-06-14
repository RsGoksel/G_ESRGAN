import os.path
import logging
import torch
from rrdbnet import RRDBNet as net
import time
import cv2
import numpy as np

import argparse 
import re

parser = argparse.ArgumentParser()
parser.add_argument('--video-path', type=str, default=" ", help='path of Video.. :)')
args = parser.parse_args()

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

vid_path = args.video_path
output_name = "outputs/out_" + vid_path.split('/')[-1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.cuda.set_device(0)
torch.cuda.empty_cache()

model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load("Weights/G_GAN.pth"), strict=True)
model.eval()

for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
torch.cuda.empty_cache()

vidcap = cv2.VideoCapture(vid_path)
current_fps = vidcap.get(cv2.CAP_PROP_FPS) 
vidcap.release()

success = True
video = cv2.VideoCapture(vid_path)

success,image = video.read()
img_L = uint2tensor4(image)
img_L = img_L.to(device)
img_E = model(img_L)
img_E = tensor2uint(img_E)

height, width, _ = img_E.shape
size = (width, height)

out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), current_fps, size)

while success:
    success,image = video.read()
    if success:
        img_L = uint2tensor4(image)
        img_L = img_L.to(device)

        img_E = model(img_L)
        img_E = tensor2uint(img_E)

        out.write(img_E)
        

video.release()
out.release()








