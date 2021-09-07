import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path


DATA_DIR = '..\\data\\'

x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_masks')
# x_train_dir = os.path.join(DATA_DIR, 'test0')
# y_train_dir = os.path.join(DATA_DIR, 'test1')

x_valid_dir = os.path.join(DATA_DIR, 'validation_images')
y_valid_dir = os.path.join(DATA_DIR, 'validation_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    # CLASSES = ['buffer-stop', 'crossing', 'guard-rail', 'train-car', 'platform',
    #             'rail', 'switch-indicator', 'switch-left', 'switch-right', 'switch-unknown',
    #             'switch-static', 'track-sign-front', 'track-signal-front', 'track-signal-back',
    #             'person-group', 'car', 'fence', 'person', 'pole', 'rail-occluder', 'truck'
    #             ]

    CLASSES = [
        'road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign',
        'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails',
        'rail-raised', 'rail-embedded'
    ]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # self.ids = os.listdir(images_dir)
        self.ids = [Path(id).stem for id in os.listdir(images_dir)]
        self.images_fps = [os.path.join(images_dir, image_id + '.jpg') for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + '.png') for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        masks = np.asarray([(mask == v) for v in self.class_values], dtype='uint8')
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)




dataset = Dataset(x_train_dir, y_train_dir, classes=['pole'])

image, mask = dataset[5] # get some sample

# mask1 = cv2.imread('..\\data\\test1\\rs00001.png', 0)
# mask2 = np.asarray([(mask1 == v) for v in class_values], dtype='uint8')

# ids = os.listdir(x_train_dir)
# ids0 = [Path(path).stem for path in ids]
# ids1 = [Path(id).stem for id in os.listdir(x_train_dir)]

# print(ids)
# print(ids0)
# print(ids1)
        



# print(mask)



visualize(
    image=image,
    cars_mask=mask.squeeze()
)