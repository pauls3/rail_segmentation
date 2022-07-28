import os
from pathlib import Path

destination = '/shared/rtis_lab/data/RailSem19/custom_split/'


def create_list():
    print("shuffling...")

    # Get file names (json and images share same names)
    arr0 = os.listdir(destination + 'validation_images')
    arr1 = []

    for i in arr0:
        arr1.append(Path(i).stem)
    
    file1 = open('rs19_val.txt', 'w')
    for ii in arr1:
        file1.write(str(destination + 'validation_images/' + ii + '.jpg' + '\t' + destination + 'validation_masks/' + ii + '.png' + '\n'))
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