import json
import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import shutil

destination = '/shared/rtis_lab/data/RailSem19/custom_split/'


def create_list():
    print("shuffling...")

    # Get file names (json and images share same names)
    arr0 = os.listdir(destination + 'validation_images')
    # arr1 = []

    # for i in arr0:
    #     arr1.append(Path(i).stem)
    
    file1 = open('rs19_val.txt', 'w')
    for ii in arr0:
        file1.write(str(destination + '/validation_images/' + ii))
    file1.close()

# def copy_paste_files(train, val):
#     print('copying and pasting files...')

#     # copy train data
#     for k in range(20):
#         i = np.random.choice(len(val))
#         shutil.copy2(destination + 'validation_images/' + str(val[i]) + '.jpg', destination + 'test_images')
#         shutil.copy2(destination + 'validation_masks/' + str(val[i]) + '.png', destination + 'test_masks')
#         shutil.copy2(destination + 'validation_jsons/' + str(val[i]) + '.json', destination + 'test_jsons')


def main():
    create_list()



if __name__== "__main__":
  main()