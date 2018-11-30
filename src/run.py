from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model 
import numpy as np
import cv2 as cv
import os



class complex_model:
    MODEL_LOC = '../nets/mixed_3_300_100_3.h5'
    TEST_DIR = '../tests/'
    DESC_FILE = '../desc/name_genres.txt'
    GENRE_COUNT = 28
    GENRE_DICT = ['Ac', 'Adu', 'Adv', 'An',
            'B', 'Co', 'Cr', 'Dr',
            'Do', 'Fam', 'Fan', 'Fi',
            'Ga', 'Hi', 'Ho', 'Mu1',
            'Mu2', 'My', 'N', 'Re',
            'Ro', 'Sc', 'Sh', 'Sp',
            'Ta', 'Th', 'Wa', 'We']

    fpd = open(DESC_FILE, 'r')
    MOVIE_DATA = fpd.readlines()

#TODO recommend some movies from top 250
#TODO make a nice looking program

    def gen_BGR_hist():
        for subdir, dirs, files in os.walk(TEST_DIR):
            x = np.empty((len(files), 512))
            for i, f in enumerate(files):
                img_path = TEST_DIR + f
                img = cv.imread(img_path)
                hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                x[i] = hist.flatten()
        return x

                
    def gen_data():
        for subdir, dirs, files in os.walk(TEST_DIR):
            x = np.empty((len(files), 224, 224, 3))
            for i, f in enumerate(files):
                img_path = TEST_DIR + f
                img = image.load_img(img_path, target_size=(224, 224))
                x[i] = image.img_to_array(img)
                x[i] = np.expand_dims(x[i], axis=0)
                x[i] = preprocess_input(x[i])
        return x


    def get_test_names():
        x = []
        for subdir, dirs, files in os.walk(TEST_DIR):
            for i, f in enumerate(files):
                x.append(f)
        return x



    def print_top_3(pred, tests):
        for f, p in zip(pred, tests):
            print(f)
            top = p.argsort()[-3:][::-1]
            for i, e in enumerate(top):
                print(i + 1, ": ", GENRE_DICT[e])


    def run_tests(model):
        tests = get_test_names()
        x1 = gen_data()
        x2 = gen_BGR_hist()
        pred = model.predict(x=[x1, x2])
        print_top_3(tests, pred)
        return pred
    
    


if __name__ == '__main__':
    model = load_model(MODEL_LOC)
    run_tests(model)
