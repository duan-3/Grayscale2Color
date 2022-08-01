from asyncio import exceptions
from tkinter import E
import cv2
import os
import glob
import numpy as np
from PIL import Image

#Image_Resizeing
files = glob.glob('./original/*.jpg')
count = 0
for f in files:
    try:
        count += 1
        img = Image.open(f)
        img_resize = img.resize((256,256))
        img_resize.save("./resized/" + str(count) + ".png")
    except OSError as e :
        pass

