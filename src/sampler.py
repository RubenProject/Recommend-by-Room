import os
import math
import cv2
import numpy as np


TRAILER_DIR = "../trailers/"
IMAGE_DIR = "../img2/"
SAMPLE_COUNT = 100
TARGET_WIDTH = 224
TARGET_HEIGHT = 224


def sample(fname):
    read_loc = TRAILER_DIR + fname
    ID = fname[:-4]
    vidcap = cv2.VideoCapture(read_loc)
    success, image = vidcap.read()

    count = 0
    s_count = 0
    succes = True
    interval = math.floor(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / SAMPLE_COUNT)
    while success:
        success, img = vidcap.read()
        if success and count % interval == 0:
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT)) 
            #norm_img = np.zeros((TARGET_WIDTH, TARGET_HEIGHT))
            #norm_img = cv2.normalize(img, norm_img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            write_loc = IMAGE_DIR + str(ID) + "_" + str(s_count) + ".PNG"
            cv2.imwrite(write_loc, img)
            s_count += 1
        count += 1
    vidcap.release()
    cv2.destroyAllWindows()


count = 1
for subdir, dirs, files in os.walk(TRAILER_DIR):
    for f in files:
        print("sampling: " + f + ": " + str(count) + "/" + str(len(files)))
        sample(f)
        count += 1
