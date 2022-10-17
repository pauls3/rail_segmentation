"""
File to shuffle RailSem19 data into 80 10 10 split for train, test, and validation sets.
Random seed: 17
"""


import json
import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import shutil

def shuffle_data():
    # print("shuffling...")

    # Get file names (json and images share same names)
    arr0 = os.listdir('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\img')
    arr1 = []

    for i in arr0:
        arr1.append(Path(i).stem)

    # with open('ai_server_splits/test_split.txt') as file:
    #     test = file.readlines()
    #     test = [line.rstrip() for line in test]
    
    # with open('ai_server_splits/train_split.txt') as file:
    #     train = file.readlines()
    #     train = [line.rstrip() for line in train]

    # with open('ai_server_splits/val_split.txt') as file:
    #     val = file.readlines()
    #     val = [line.rstrip() for line in val]
        
    random.seed(17)
    random.shuffle(arr1)

    # Train = 80%
    # Validation = 10%
    # Test = 10%

    # train = arr1[:int((len(arr1)+1)*.80)] #Remaining 80% to training set
    # rest_data = arr1[int((len(arr1)+1)*.80):] #Splits 20% data to test set
    # val =  rest_data[int((len(arr1)+1)*.50):] # Split 50% (10% overall)
    # test = rest_data[:int((len(arr1)+1)*.50)] # split 50% (10% overall)

    train = arr1[:int((len(arr1)+1)*.70)] #Remaining 80% to training set
    rest_data = arr1[int((len(arr1)+1)*.70):] #Splits 20% data to test set
    val =  rest_data[int((len(arr1)+1)*.66):] # Split 50% (10% overall)
    test = rest_data[:int((len(arr1)+1)*.33)] # split 50% (10% overall)

    train, val, test = np.split(arr1, [int(.7*len(arr1)), int(.9*len(arr1))])

    print(len(arr1))
    print(len(train))
    print(len(val))
    print(len(test))


    # if check_duplicates(train, val, test):
    #     print('There are duplicates in the data splits!')
    # else:
    #     copy_paste_files(train, val, test)

    # train = arr1[:int((len(arr1)+1)*.80)] #Remaining 80% to training set
    # val = arr1[int((len(arr1)+1)*.80):] #Splits 20% data to test set

    create_directories()
    copy_paste_files(train, val, test)


"""
Check if there are any duplicates in the data splits. Returns false if there are none, returns true if there are.
"""

def check_duplicates(train, val, test):
    for i in train:
        for j in val:
            if i == j:
                return True
    
        for j in test:
            if i == j:
                return True
    
    for i in val:
        for j in test:
            if i == j:
                return True
    
    return False



def create_directories(): 
    dirs = [ 
        'train_images', 'train_masks', 
        'validation_images', 'validation_masks', 
        'test_images', 'test_masks' 
        ] 
 
    parent = 'C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\custom_split' 
    # os.mkdir(parent) 
 
    for ii in dirs: 
         os.mkdir(os.path.join(parent, ii)) 



def copy_paste_files(train, val, test):
    print('copying and pasting files...')
    destination = 'C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\custom_split\\'



    # copy train data
    for i in train:
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\img\\' + i + '.png', destination + 'train_images')
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\masks\\' + i + '.png', destination + 'train_masks')
        # shutil.copy2('/home/paul/data/RailSem19/jsons/rs19_val/' + i + '.json', destination + 'train_jsons')

    
    # copy validation data
    for i in val:
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\img\\' + i + '.png', destination + 'validation_images')
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\masks\\' + i + '.png', destination + 'validation_masks')
        # shutil.copy2('/home/paul/data/RailSem19/jsons/rs19_val/' + i + '.json', destination + 'validation_jsons')

    # copy test data
    for i in test:
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\img\\' + i + '.png', destination + 'test_images')
        shutil.copy2('C:\\Users\\Paul\\Desktop\\unlv\\thesis\\dataset\\rtis-rail-2022\\masks\\' + i + '.png', destination + 'test_masks')
        # shutil.copy2('/home/paul/data/RailSem19/jsons/rs19_val/' + i + '.json', destination + 'test_jsons')



def main():
    shuffle_data()


if __name__== "__main__":
  main()