import json
import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import shutil

destination = '../data/'


def shuffle_data():
    print("shuffling...")

    # Get file names (json and images share same names)
    arr0 = os.listdir(destination + 'validation_images')
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
    val = arr1[int((len(arr1)+1)*.98):] #Splits 20% data to test set

    copy_paste_files(train, val)


def copy_paste_files(train, val):
    print('copying and pasting files...')

    # copy train data
    for k in range(20):
        i = np.random.choice(len(val))
        shutil.copy2(destination + 'validation_images/' + str(val[i]) + '.jpg', destination + 'test_images')
        shutil.copy2(destination + 'validation_masks/' + str(val[i]) + '.png', destination + 'test_masks')
        shutil.copy2(destination + 'validation_jsons/' + str(val[i]) + '.json', destination + 'test_jsons')


def main():
    shuffle_data()



if __name__== "__main__":
  main()