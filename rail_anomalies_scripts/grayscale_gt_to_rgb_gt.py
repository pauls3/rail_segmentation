import cv2
import numpy as np
import os
from pathlib import Path

masks_machine_dir = Path("rad_9_22_2026\\masks_machine")
#masks_machine_dir = Path("rad_9_22_2026\\masks_machine")
target_dir = "rad_9_22_2026\\masks_machine_to_rgb"
file_type = ".png"


for file_path in masks_machine_dir.iterdir():
    if file_path.is_file() and file_path.suffix.lower() == file_type:
        im = cv2.imread(file_path)

        ## Convert grayscale to BGR (not RGB)
        im[np.all(im == (0, 0, 0), axis=-1)] = (60,20,220)          ## person
        im[np.all(im == (1, 1, 1), axis=-1)] = (69,0,0)             ## truck
        im[np.all(im == (2, 2, 2), axis=-1)] = (139,149,230)        ## rail-track
        im[np.all(im == (3, 3, 3), axis=-1)] = (34,142,106)         ## vegetation-overgrowth
        im[np.all(im == (4, 4, 4), axis=-1)] = (142,0,0)            ## car
        im[np.all(im == (5, 5, 5), axis=-1)] = (100,80,0)           ## on-rails
        im[np.all(im == (6, 6, 6), axis=-1)] = (0,220,220)          ## traffic-sign
        im[np.all(im == (7, 7, 7), axis=-1)] = (128,64,128)         ## road
        im[np.all(im == (8, 8, 8), axis=-1)] = (232,35,244)         ## sidewalk
        im[np.all(im == (9, 9, 9), axis=-1)] = (70,70,70)           ## construction
        im[np.all(im == (10, 10, 10), axis=-1)] = (128,0,92)        ## tram-track
        im[np.all(im == (11, 11, 11), axis=-1)] = (153,153,153)     ## pole
        im[np.all(im == (12, 12, 12), axis=-1)] = (30,170,250)      ## traffic-light
        im[np.all(im == (13, 13, 13), axis=-1)] = (0,255,255)       ## mud-pumping
        im[np.all(im == (14, 14, 14), axis=-1)] = (153,153,190)     ## fence
        im[np.all(im == (15, 15, 15), axis=-1)] = (152,251,152)     ## terrain
        im[np.all(im == (16, 16, 16), axis=-1)] = (180,130,70)      ## sky
        im[np.all(im == (17, 17, 17), axis=-1)] = (0,68,63)         ## rail-embedded
        im[np.all(im == (18, 18, 18), axis=-1)] = (255,255,0)       ## rail-raised
        im[np.all(im == (19, 19, 19), axis=-1)] = (41,40,90)        ## track-bed
        im[np.all(im == (20, 20, 20), axis=-1)] = (0,0,156)         ## standing-water

        cv2.imwrite(os.path.join(target_dir, file_path.name),im)
