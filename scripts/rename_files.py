import os

folder = 'C:/Users/Paul/source/datasets/rail_semantic/rail_segmentation/'
count = 0
# count increase by 1 in each iteration
# iterate all files from a directory
for file_name in os.listdir(folder):

    if count < 10:
        strCount = "000" + str(count)
    elif count < 100:
        strCount = "00" + str(count)
    elif count < 1000:
        strCount = "0" + str(count)
    else:
        strCount = str(count)


    # Construct old file name
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + strCount + ".png"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(folder)
print(res)

print(count)