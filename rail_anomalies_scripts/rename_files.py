import os

image_dir = "rad_9_22_2026\\img"
masks_human_dir = "rad_9_22_2026\\masks_human"
masks_machine_dir = "rad_9_22_2026\\masks_machine"
masks_machine_rgb_dir = "rad_9_22_2026\\masks_machine_to_rgb"
json_dir = "rad_9_22_2026\\jsons"

count = 0

# count increase by 1 in each iteration
# iterate all files from a directory
for file_name in os.listdir(image_dir):

    if count < 10:
        strCount = "000" + str(count)
    elif count < 100:
        strCount = "00" + str(count)
    elif count < 1000:
        strCount = "0" + str(count)
    else:
        strCount = str(count)

    split_file_name = os.path.splitext(file_name)

    # Construct old file name    
    source_file_img = os.path.join(image_dir, file_name)
    source_file_human = os.path.join(masks_human_dir, split_file_name[0] + ".png")
    source_file_machine = os.path.join(masks_machine_dir, split_file_name[0] + ".png")
    source_file_machine_rgb = os.path.join(masks_machine_rgb_dir, split_file_name[0] + ".png")
    source_file_json = os.path.join(json_dir, file_name + ".json")


    split_img = os.path.splitext(source_file_img)
    file_extension_img = split_img[1]

    split_human = os.path.splitext(source_file_human)
    file_extension_human = split_human[1]

    split_machine = os.path.splitext(source_file_machine)
    file_extension_machine = split_machine[1]

    split_machine_rgb = os.path.splitext(source_file_machine_rgb)
    file_extension_machine_rgb = split_machine_rgb[1]


    destination_img = os.path.join(image_dir, strCount + file_extension_img)
    destination_human = os.path.join(masks_human_dir, strCount + file_extension_human)
    destination_machine = os.path.join(masks_machine_dir, strCount + file_extension_machine)    
    destination_machine_rgb = os.path.join(masks_machine_rgb_dir, strCount + file_extension_machine_rgb)    
    destination_json = os.path.join(json_dir, strCount + ".json")


    # Renaming the file
    os.rename(source_file_img, destination_img)
    os.rename(source_file_human, destination_human)
    os.rename(source_file_machine, destination_machine)
    os.rename(source_file_machine_rgb, destination_machine_rgb)
    os.rename(source_file_json, destination_json)


    count += 1


print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(image_dir)
print(res)

print(count)