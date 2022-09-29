import cv2
import numpy as np

# Load image
im = cv2.imread('C:/Users/Paul/source/datasets/rs19/uint8/rs19_val/rs04522.png')

# Make all perfectly green pixels white
im[np.all(im == (0, 0, 0), axis=-1)] = (128,64,128)
im[np.all(im == (1, 1, 1), axis=-1)] = (232,35,244)
im[np.all(im == (2, 2, 2), axis=-1)] = (70,70,70)
im[np.all(im == (3, 3, 3), axis=-1)] = (128,0,192)
im[np.all(im == (4, 4, 4), axis=-1)] = (153,153,190)
im[np.all(im == (5, 5, 5), axis=-1)] = (153,153,153)
im[np.all(im == (6, 6, 6), axis=-1)] = (30,170,250)
im[np.all(im == (7, 7, 7), axis=-1)] = (0,220,220)
im[np.all(im == (8, 8, 8), axis=-1)] = (35,142,107)
im[np.all(im == (9, 9, 9), axis=-1)] = (152,251,152)
im[np.all(im == (10, 10, 10), axis=-1)] = (180,130,70)
im[np.all(im == (11, 11, 11), axis=-1)] = (60,20,220)
im[np.all(im == (12, 12, 12), axis=-1)] = (140,150,230)
im[np.all(im == (13, 13, 13), axis=-1)] = (142,0,0)
im[np.all(im == (14, 14, 14), axis=-1)] = (70,0,0)
im[np.all(im == (15, 15, 15), axis=-1)] = (40,40,90)
im[np.all(im == (16, 16, 16), axis=-1)] = (100,80,0)
im[np.all(im == (17, 17, 17), axis=-1)] = (254,254,0)
im[np.all(im == (18, 18, 18), axis=-1)] = (63,68,0)


# Save result
cv2.imwrite('C:/Users/Paul/source/datasets/rs19/rgb-ground-truth/rs04522.png',im)