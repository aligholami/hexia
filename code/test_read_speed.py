import sys
import numpy as np
import os
import cv2

PATH_TO_IMAGES = '../data/train/images/full-image-dir/'


image_list = os.listdir(PATH_TO_IMAGES)
print(image_list)
loaded_to_mem = [] 
loaded_to_mem_size = 0

for idx, image_name in enumerate(image_list):
    print("Loaded {} MB into memory".format(loaded_to_mem_size))
    img = cv2.imread(PATH_TO_IMAGES + image_name)
    array_size_in_mb = sys.getsizeof(img) / float(1 << 20)
    loaded_to_mem_size += array_size_in_mb
    loaded_to_mem.append(img)

