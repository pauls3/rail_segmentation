import numpy as np
from PIL import Image
import os
from pathlib import Path


image_dir = Path("brendan_supervisely\\new_masks_machine")
file_type = ".png"

for file_path in image_dir.iterdir():
    if file_path.is_file() and file_path.suffix.lower() == file_type:
        #print(f"{file_path}")
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)


        """
        ## Paul's new class id 
        modified_array = np.where(img_array == 6, 255, img_array)
        modified_array = np.where(modified_array == 7, 6, modified_array)
        modified_array = np.where(modified_array == 8, 7, modified_array)
        modified_array = np.where(modified_array == 9, 8, modified_array)
        modified_array = np.where(modified_array == 10, 9, modified_array)
        modified_array = np.where(modified_array == 11, 10, modified_array)
        modified_array = np.where(modified_array == 12, 11, modified_array)
        modified_array = np.where(modified_array == 13, 12, modified_array)
        modified_array = np.where(modified_array == 14, 13, modified_array)
        modified_array = np.where(modified_array == 15, 14, modified_array)
        modified_array = np.where(modified_array == 16, 15, modified_array)
        modified_array = np.where(modified_array == 17, 16, modified_array)
        modified_array = np.where(modified_array == 18, 17, modified_array)
        modified_array = np.where(modified_array == 19, 18, modified_array)
        modified_array = np.where(modified_array == 20, 19, modified_array)
        modified_array = np.where(modified_array == 21, 20, modified_array)
        modified_array = np.where(modified_array == 22, 21, modified_array)
        """

        """
        ## Brendan's class id conversion
        modified_array = np.where(img_array == 14, 255, img_array)
        modified_array = np.where(modified_array == 1, 254, modified_array)
        modified_array = np.where(modified_array == 2, 253, modified_array)
        modified_array = np.where(modified_array == 3, 252, modified_array)
        modified_array = np.where(modified_array == 4, 251, modified_array)
        modified_array = np.where(modified_array == 5, 250, modified_array)
        modified_array = np.where(modified_array == 6, 249, modified_array)
        modified_array = np.where(modified_array == 7, 248, modified_array)
        modified_array = np.where(modified_array == 8, 247, modified_array)
        modified_array = np.where(modified_array == 9, 246, modified_array)
        modified_array = np.where(modified_array == 10, 245, modified_array)
        modified_array = np.where(modified_array == 11, 244, modified_array)
        modified_array = np.where(modified_array == 12, 243, modified_array)
        modified_array = np.where(modified_array == 13, 242, modified_array)
        modified_array = np.where(modified_array == 15, 241, modified_array)
        modified_array = np.where(modified_array == 16, 240, modified_array)
        modified_array = np.where(modified_array == 17, 239, modified_array)
        modified_array = np.where(modified_array == 18, 238, modified_array)
        modified_array = np.where(modified_array == 19, 237, modified_array)
        modified_array = np.where(modified_array == 20, 236, modified_array)
        modified_array = np.where(modified_array == 21, 235, modified_array)
        modified_array = np.where(modified_array == 22, 234, modified_array)
        """

        modified_array = np.where(img_array == 254, 7, img_array)
        modified_array = np.where(modified_array == 253, 15, modified_array)
        modified_array = np.where(modified_array == 252, 16, modified_array)
        modified_array = np.where(modified_array == 251, 17, modified_array)
        modified_array = np.where(modified_array == 250, 18, modified_array)
        modified_array = np.where(modified_array == 249, 19, modified_array)
        modified_array = np.where(modified_array == 248, 20, modified_array)
        modified_array = np.where(modified_array == 247, 21, modified_array)
        modified_array = np.where(modified_array == 246, 22, modified_array)
        modified_array = np.where(modified_array == 245, 2, modified_array)
        modified_array = np.where(modified_array == 244, 3, modified_array)
        modified_array = np.where(modified_array == 243, 4, modified_array)
        modified_array = np.where(modified_array == 242, 5, modified_array)
        modified_array = np.where(modified_array == 241, 1, modified_array)
        modified_array = np.where(modified_array == 240, 8, modified_array)
        modified_array = np.where(modified_array == 239, 9, modified_array)
        modified_array = np.where(modified_array == 238, 10, modified_array)
        modified_array = np.where(modified_array == 237, 11, modified_array)
        modified_array = np.where(modified_array == 236, 12, modified_array)
        modified_array = np.where(modified_array == 235, 13, modified_array)
        modified_array = np.where(modified_array == 234, 14, modified_array)

        new_img = Image.fromarray(modified_array.astype(np.uint8), "L")
        new_img.save(os.path.join("brendan_supervisely\\new_masks_machine_1", file_path.name))



"""
# 1. Load the grayscale image (mode 'L' ensures it's single-channel grayscale)
img = Image.open("128.png").convert("L")

# 2. Convert the image into a NumPy array
img_array = np.array(img)

# --- Option B: Conditional changes ---
# Example: If a pixel value is less than 128, set it to 0 (Thresholding)
modified_array = np.where(img_array == 5, 255, img_array)


# 3. Convert back to a Pillow Image and save
new_img = Image.fromarray(modified_array.astype(np.uint8), "L")
new_img.save("128_modified.png")
"""


