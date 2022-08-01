import cv2
import os
import glob
import numpy as np
from PIL import Image

#Image_GrayScaling
count = 0
for count in range(1,4320):
    try:
        img = Image.open("./resized/" + str(count) + ".png").convert("L")
        img_numpy = np.array(img,'uint8')
        cv2.imwrite("./gray/" + str(count) + ".png", img_numpy)
        count += 1
    except OSError as e :
        print(count)
        pass