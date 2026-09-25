import json
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
import sys

class_to_id = {
    "person": 0,
    "truck": 1,
    "rail-track": 2,
    "vegetation-overgrowth": 3,
    "car": 4,
    "on-rails": 5,
    "traffic-sign": 6,
    "road": 7,
    "sidewalk": 8,
    "construction": 9,
    "tram-track": 10,
    "pole": 11,
    "traffic-light": 12,
    "mud-pumping": 13,
    "fence": 14,
    "terrain": 15,
    "sky": 16,
    "rail-embedded": 17,
    "rail-raised": 18,
    "trackbed": 19,
    "standing-water":20,
    "void": 255
}

json_dir = Path("rad_9_22_2026\\jsons")
json_ext = ".json"
mask_dir = "rad_9_22_2026\\masks_machine"

for file_path in json_dir.iterdir():
    if file_path.is_file() and file_path.suffix.lower() == json_ext:
        with open(file_path) as f:
            json_info = json.load(f)

            ## Get height and width
            height = json_info["size"]["height"]
            width = json_info["size"]["width"]

            ## Setup the canvas for drawing
            image = Image.new("L", (width, height), "black")
            draw = ImageDraw.Draw(image)

            ## Iterate all the polygon objects and draw them with classID fill
            for polygon in json_info["objects"]:

                ## Get fill value
                classTitle = polygon["classTitle"]
                classID = class_to_id[classTitle]

                ## Draw the polygon
                points_tuple = tuple(polygon["points"]["exterior"])
                draw.polygon(points_tuple, fill=classID, width=1)
            
            ## Save image
            image.save(os.path.join(mask_dir, file_path.name.split('.')[0] + ".png"))
