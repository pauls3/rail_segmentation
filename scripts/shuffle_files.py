import json
import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import shutil


def shuffle_data():
    print("shuffling...")

    # Get file names (json and images share same names)
    arr0 = os.listdir('../RailSem19/jpgs/rs19_val')
    arr1 = []

    for i in arr0:
        arr1.append(Path(i).stem)
    
    random.shuffle(arr1)

    # Train = 80%
    # Validation = 10%
    # Test = 10%
    train, val, test = np.split(arr1, [int(.8*len(arr1)), int(.9*len(arr1))])
    copy_paste_files(train, val, test)
    


def copy_paste_files(train, val, test):
    print('copying and pasting files...')
    destination = '../data/'

    # copy train data
    for i in train:
        shutil.copy2('../RailSem19/jpgs/rs19_val/' + i + '.jpg', destination + 'train_images')
        shutil.copy2('../RailSem19/uint8/rs19_val/' + i + '.png', destination + 'train_masks')
        shutil.copy2('../RailSem19/jsons/rs19_val/' + i + '.json', destination + 'train_jsons')

    
    # copy validation data
    for i in val:
        shutil.copy2('../RailSem19/jpgs/rs19_val/' + i + '.jpg', destination + 'validation_images')
        shutil.copy2('../RailSem19/uint8/rs19_val/' + i + '.png', destination + 'validation_masks')
        shutil.copy2('../RailSem19/jsons/rs19_val/' + i + '.json', destination + 'validation_jsons')

    # copy test data
    for i in test:
        shutil.copy2('../RailSem19/jpgs/rs19_val/' + i + '.jpg', destination + 'test_images')
        shutil.copy2('../RailSem19/uint8/rs19_val/' + i + '.png', destination + 'test_masks')
        shutil.copy2('../RailSem19/jsons/rs19_val/' + i + '.json', destination + 'test_jsons')



def main():
    shuffle_data()



if __name__== "__main__":
  main()