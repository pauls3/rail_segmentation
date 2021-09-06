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
    arr0 = os.listdir('../data/RailSem19/rs19_val/jpgs/rs19_val')
    arr1 = []

    for i in arr0:
        arr1.append(Path(i).stem)
    
    random.seed(21)
    random.shuffle(arr1)

    # Train = 80%
    # Validation = 20%
    ## Test = 10%
    # train, val, test = np.split(arr1, [int(.8*len(arr1)), int(.9*len(arr1))])
    train = arr1[:int((len(arr1)+1)*.80)] #Remaining 80% to training set
    val = arr1[int((len(arr1)+1)*.80):] #Splits 20% data to test set

    copy_paste_files(train, val)


def copy_paste_files(train, val):
    print('copying and pasting files...')
    destination = '../data/'

    # copy train data
    for i in train:
        shutil.copy2('../data/RailSem19/rs19_val/jpgs/rs19_val/' + i + '.jpg', destination + 'train_images')
        shutil.copy2('../data/RailSem19/rs19_val/uint8/rs19_val/' + i + '.png', destination + 'train_masks')
        shutil.copy2('../data/RailSem19/rs19_val/jsons/rs19_val/' + i + '.json', destination + 'train_jsons')

    
    # copy validation data
    for i in val:
        shutil.copy2('../data/RailSem19/rs19_val/jpgs/rs19_val/' + i + '.jpg', destination + 'validation_images')
        shutil.copy2('../data/RailSem19/rs19_val/uint8/rs19_val/' + i + '.png', destination + 'validation_masks')
        shutil.copy2('../data/RailSem19/rs19_val/jsons/rs19_val/' + i + '.json', destination + 'validation_jsons')

    # copy test data
    # for i in test:
    #     shutil.copy2('../data/RailSem19/rs19_val/jpgs/rs19_val/' + i + '.jpg', destination + 'test_images')
    #     shutil.copy2('../data/RailSem19/rs19_val/uint8/rs19_val/' + i + '.png', destination + 'test_masks')
    #     shutil.copy2('../data/RailSem19/rs19_val/jsons/rs19_val/' + i + '.json', destination + 'test_jsons')



def main():
    shuffle_data()



if __name__== "__main__":
  main()