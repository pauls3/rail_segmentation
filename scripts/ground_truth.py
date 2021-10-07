import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path
import albumentations as albu
import segmentation_models_pytorch as smp
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image


np.set_printoptions(threshold=sys.maxsize)


DATA_DIR = '../data/'

x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_masks')

x_valid_dir = os.path.join(DATA_DIR, 'validation_images')
y_valid_dir = os.path.join(DATA_DIR, 'validation_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')



colors = [
	(128, 64, 128),
	(244, 35, 232),
	(70, 70, 70),
	(192, 0, 128),
	(190, 153, 153),
	(153, 153, 153),
	(250, 170, 30),
	(220, 220, 0),
	(107, 142, 35),
	(152, 251, 152),
	(70, 130, 180),
	(220, 20, 60),
	(230, 150, 140),
	(0, 0, 142),
	(0, 0, 70),
	(90, 40, 40),
	(0, 80, 100),
	(0, 254, 254),
	(0, 68, 63)
]




arr0 = os.listdir('../data/test_masks')
arr1 = []

for i in arr0:
    arr1.append(Path(i).stem)

print(arr1)


for i in range(0, len(arr1)):
    
    image_vis = cv2.imread('../data/test_masks/' + arr1[i] + '.png')

    img_to_draw = Image.fromarray(image_vis)
    draw = ImageDraw.Draw(img_to_draw)

    for j in range(len(image_vis)):
        for l in range(len(image_vis[j])):
            if(image_vis[j][l][0] != 255):
                draw.point((l, j), fill=colors[image_vis[j][l][0]])
            else:
                draw.point((l, j), fill=(255, 255, 255))

    img_to_draw.save('./exp_0/ground_truth/' + arr1[i] + '.png')

