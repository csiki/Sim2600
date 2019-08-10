import cv2
import numpy as np
from glob import glob
import os, shutil
import matplotlib.pyplot as plt


outpath = 'outFrames_fixed/'
img_paths = glob('outFrames/*.png')

shutil.rmtree(outpath, False)
os.mkdir(outpath)

purple = np.array([172, 125, 214], dtype=np.uint8)
tofind = np.zeros((8, 3*4, 3), dtype=np.uint8)  # pause sign
tofind[:, :4] = tofind[:, 8:] = purple
desired_y_pos = 20  # has to be above (less than) of all the possible real values
desired_height = 175

for img_path in img_paths:
    img = cv2.imread(img_path)
    
    match = cv2.matchTemplate(img, tofind, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = min_loc
    img = img[top_left[1] - desired_y_pos:top_left[1]+desired_height, :, :]
    
    cv2.imwrite(outpath + os.path.basename(img_path), img)

print('done')
