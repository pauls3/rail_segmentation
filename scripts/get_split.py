"""
Get file names for split (train/val/test) in AI server.
"""


import os
from pathlib import Path


def get_names():
    print("shuffling...")

    # Get file names from image directories
    test = os.listdir('/home/stanik/rtis_lab/data/RailSem19/custom_split/test_images')
    val = os.listdir('/home/stanik/rtis_lab/data/RailSem19/custom_split/validation_images')
    train = os.listdir('/home/stanik/rtis_lab/data/RailSem19/custom_split/train_images')

    # Remove the file extensions
    test0 = []
    val0 = []
    train0 = []
    for ii in test:
        test0.append(Path(ii).stem)
    for ii in val:
        val0.append(Path(ii).stem)
    for ii in train:
        train0.append(Path(ii).stem)
    
    os.mkdir('splits')

    with open('splits/test_split.txt', 'w') as f:
        f.write('\n'.join([' '.join(text) for text in test0]))
    with open('splits/train_split.txt', 'w') as f:
        f.write('\n'.join([' '.join(text) for text in val0]))
    with open('splits/val_split.txt', 'w') as f:
        f.write('\n'.join([' '.join(text) for text in train0]))


def main():
    get_names()


if __name__== "__main__":
  main()