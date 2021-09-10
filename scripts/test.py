import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path
import albumentations as albu
import segmentation_models_pytorch as smp



DATA_DIR = '../data/'

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




def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.GaussNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.IAASharpen(p=1),
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)




def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)








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



# visualize(
#     image=image,
#     rail_mask=mask.squeeze()
# )





ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [
    'road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign',
    'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails',
    'rail-raised', 'rail-embedded'
]
ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.PSPNet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)




preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Training dataset
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# Validation dataset
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=False, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)




# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
# metrics = [
#     smp.utils.metrics.IoU(threshold=0.5),
# ]
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
]



optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])



# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)



valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# --------------------------------------------
# Included names 

classNames = [
    'road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign',
    'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails',
    'rail-raised', 'rail-embedded'
]



train_epoch.metrics[0].__name__ = 'iou_score'
for i in range(1, 19):
    train_epoch.metrics[i].__name__ = classNames[i - 1]

valid_epoch.metrics[0].__name__ = 'iou_score'
for i in range(1, 19):
    valid_epoch.metrics[i].__name__ = classNames[i - 1]

# --------------------------------------------


state_dict = torch.load('./best_model.pth')
model.load_state_dict(state_dict.state_dict())


# train model for 200 epochs

max_score = 0

for i in range(0, 200):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    # if max_score < valid_logs['iou_score']:
    #     max_score = valid_logs['iou_score']
    #     torch.save(model, './best_model.pth')
    #     print('Model saved!')

    print(train_logs)
    print(valid_logs)

    torch.save(model, './checkpoints/epoch_' + str(i) +'.pth')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')



# load best saved checkpoint
# best_model = torch.load('./best_model.pth')