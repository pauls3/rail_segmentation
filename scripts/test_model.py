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
# x_train_dir = os.path.join(DATA_DIR, 'test0')
# y_train_dir = os.path.join(DATA_DIR, 'test1')

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





metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

loss = smp.utils.losses.DiceLoss()


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




ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [
    'road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign',
    'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails',
    'rail-raised', 'rail-embedded'
]
ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# load best saved checkpoint
best_model = torch.load('./checkpoints/epoch_36.pth')

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=False, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)



# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)



arr0 = os.listdir('../data/test_images')
arr1 = []

for i in arr0:
    arr1.append(Path(i).stem)

print(arr1)


for i in range(0, len(arr1)):
    
    # image_vis = test_dataset_vis[i][0].astype('uint8')
    image_vis = read_image('../data/test_images/' + arr1[i] + '.jpg')
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    print(pr_mask.shape)



    empty_image = torch.zeros([3, pr_mask.shape[1], pr_mask.shape[2]], dtype=torch.uint8)

    # test_image = draw_segmentation_masks(empty_image, masks=pr_mask, alpha=0.7, colors=colors)
    ndarr = empty_image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)

    for k in range(len(pr_mask)):
        for j in range(len(pr_mask[k])):
            for l in range(len(pr_mask[k][j])):
                if(pr_mask[k][j][l]):
                    draw.point((l, j), fill=colors[k])
                # else:
                # 	draw.point((j, l))

    # plt.show(draw)
    # plt.show(img_to_draw)
    img_to_draw.save('./exp_0/predicted/' + arr1[i] + '.png')








    # visualize(
    #     image=image_vis, 
    #     ground_truth_mask=gt_mask, 
    #     # predicted_mask=pr_mask
    #     predicted_mask=test_image
    # )