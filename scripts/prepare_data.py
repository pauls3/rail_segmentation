import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision.io import read_image
import json
from torch.utils.data import DataLoader, IterableDataset
import bisect
from skimage import io, transform
import numpy as np


# def SingeJsonDataset():
#     # implement a single json dataset here...

#     list_of_datasets = []
#     for j in os.path.listdir(root_dir):
#         if not j.endswith('.json'):
#             continue  # skip non-json files
#         list_of_datasets.append(SingeJsonDataset(json_file=j, root_dir=root_dir, transform=None))
#     # once all single json datasets are created you can concat them into a single one:
#     multiple_json_dataset = data.ConcatDataset(list_of_datasets)

DATA_DIR = '../data/'

train_images_dir = os.path.join(DATA_DIR, 'train_images')
train_jsons_dir = os.path.join(DATA_DIR, 'train_jsons')
train_masks_dir = os.path.join(DATA_DIR, 'train_masks')

val_images_dir = os.path.join(DATA_DIR, 'val_images')
val_jsons_dir = os.path.join(DATA_DIR, 'val_jsons')
val_masks_dir = os.path.join(DATA_DIR, 'val_masks')

test_images_dir = os.path.join(DATA_DIR, 'test_images')
test_jsons_dir = os.path.join(DATA_DIR, 'test_jsons')
test_masks_dir = os.path.join(DATA_DIR, 'test_masks')


class MyDataLoader(data.Dataset):
    def __init__(self, path_to_imgs, path_to_json, augmentation=None, preprocessing=None):
        self.path_to_images = path_to_imgs
        self.path_to_json = path_to_json
        self.image_ids = os.listdir(path_to_imgs)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mapping = {"buffer-stop": (70,70,70),
                        "crossing": (128,64,128),
                        "guard-rail": (0,255,0),
                        "train-car" :  (100,80,0),
                        "platform" : (232,35,244),
                        "rail": (255,255,0),
                        "switch-indicator": (127,255,0),
                        "switch-left": (255,255,0),
                        "switch-right": (127,127,0),
                        "switch-unknown": (191,191,0),
                        "switch-static": (0,255,127),
                        "track-sign-front" : (0,220,220),
                        "track-signal-front" : (30,170,250),
                        "track-signal-back" : (0,85,125),
                        #rail occluders
                        "person-group" : (60,20,220),
                        "car" : (142,0,0),
                        "fence" : (153,153,190),
                        "person" : (60,20,220),
                        "pole" : (153,153,153),
                        "rail-occluder" : (255,255,255),
                        "truck" : (70,0,0)
                        }

        # returns bounding boxes, polygons, polyline-pair, and polyline from the json file
    def get_masks(jsonPath):
        inp_json = json.load(open(jsonPath, 'r'))
        # masks = 
        # for obj in inp_json["objects"]:
        return inp_json["objects"]
        

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = io.imread(img_id)
        masks = get_masks(self.path_to_json, img_id + ".json")

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=img, mask=masks)
        #     image, mask = sample['image'], sample['mask']
        
        # # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=img, mask=masks)
        #     image, mask = sample['image'], sample['mask']

        return img, masks






    def mask_to_class_rgb(self, mask):
        print('----mask->rgb----')
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        print('unique values rgb    ', torch.unique(mask)) 
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3
        print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])
       
        return mask_out
