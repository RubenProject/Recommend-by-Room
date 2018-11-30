from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model 
import numpy as np
import cv2 as cv
import os
import random

MODEL_LOC = '../nets/mixed_3_300_100_3.h5'
TEST_DIR = '../tests/'


class Complex_model:
    moviedata_file = '../desc/name_genres.txt'
    id_genre_map = ['Ac', 'Adu', 'Adv', 'An',
            'B', 'Co', 'Cr', 'Dr',
            'Do', 'Fam', 'Fan', 'Fi',
            'Ga', 'Hi', 'Ho', 'Mu1',
            'Mu2', 'My', 'N', 'Re',
            'Ro', 'Sc', 'Sh', 'Sp',
            'Ta', 'Th', 'Wa', 'We']

    
#TODO recommend some movies from top 250
    def __init__ (self, model_loc):
        fpd = open(self.moviedata_file, 'r')
        self.movie_db = fpd.readlines()
        self.model = load_model(model_loc)


    def gen_BGR_hist_batch(self, test_loc):
        for subdir, dirs, files in os.walk(test_loc):
            x = np.empty((len(files), 512))
            for i, f in enumerate(files):
                img_path = test_loc + f
                img = cv.imread(img_path)
                hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                x[i] = hist.flatten()
        return x

                
    def gen_data_batch(self, test_loc):
        for subdir, dirs, files in os.walk(test_loc):
            x = np.empty((len(files), 224, 224, 3))
            for i, f in enumerate(files):
                img_path = test_loc + f
                img = image.load_img(img_path, target_size=(224, 224))
                x[i] = image.img_to_array(img)
                x[i] = np.expand_dims(x[i], axis=0)
                x[i] = preprocess_input(x[i])
        return x


    def get_names_batch(self, test_loc):
        x = []
        for subdir, dirs, files in os.walk(test_loc):
            for i, f in enumerate(files):
                x.append(f)
        return x



    def print_top_n_batch(self, pred, tests, n):
        if n > len(pred):
            return
        for f, p in zip(pred, tests):
            print(f)
            top = p.argsort()[-n:][::-1]
            for i, e in enumerate(top):
                print(i + 1, ": ", self.id_genre_map[e])


    def run_tests(self, test_loc):
        tests = self.get_names_batch(test_loc)
        x1 = self.gen_data_batch(test_loc)
        x2 = self.gen_BGR_hist_batch(test_loc)
        pred = self.model.predict(x=[x1, x2])
        self.print_top_n_batch(tests, pred, 3)


    def gen_BGR_hist(self, img_path):
        x = np.empty((1, 512))
        img = cv.imread(img_path)
        hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        x[0] = hist.flatten()
        return x


    def gen_data(self, img_path):
        x = np.empty((1, 224, 224, 3))
        img = image.load_img(img_path, target_size=(224, 224))
        x[0] = image.img_to_array(img)
        x[0] = np.expand_dims(x[0], axis=0)
        x[0] = preprocess_input(x[0])
        return x


    def get_top_n(self, pred, n):
        if n > len(pred):
            return None
        top = pred.argsort()[-n:][::-1]
        top_n = []
        for e in top:
            top_n.append(self.id_genre_map[e])
        return top_n


    def run_test(self, img):
        x1 = self.gen_data(img)
        x2 = self.gen_BGR_hist(img)
        pred = self.model.predict(x=[x1, x2])
        print(pred)
        top = self.get_top_n(pred[0], 3)
        return top


    def get_rec(self, genres):
        if len(genres) != 3:
            return ''
        c_id1 = []
        c_id2 = []
        c_id3 = []
        for line in self.movie_db:
            s = line.split('|')
            s_g = s[2].split(' ')
            for g in s_g:
                if g == genres[0]:
                    c_id1.append(int(s[0]))
                if g == genres[1]:
                    c_id2.append(int(s[0]))
                if g == genres[2]:
                    c_id3.append(int(s[0]))
        c_id12 = list(set(c_id1) & set(c_id2))
        c_id13 = list(set(c_id1) & set(c_id3))
        c_id23 = list(set(c_id2) & set(c_id3))
        c_id123 = list(set(c_id12) & set(c_id3))
        c_id = -1
        if len(c_id123) != 0:
            r_id = random.randint(0, len(c_id123) - 1)
            c_id = c_id123[r_id]
        elif len(c_id12) != 0:
            r_id = random.randint(0, len(c_id12) - 1)
            c_id = c_id12[r_id]
        elif len(c_id13) != 0:
            r_id = random.randint(0, len(c_id13) - 1)
            c_id = c_id13[r_id]
        elif len(c_id23) != 0:
            r_id = random.randint(0, len(c_id23) - 1)
            c_id = c_id23[r_id]
        elif len(c_id1) != 0:
            r_id = random.randint(0, len(c_id1) - 1)
            c_id = c_id1[r_id]
        elif len(c_id2) != 0:
            r_id = random.randint(0, len(c_id2) - 1)
            c_id = c_id2[r_id]
        elif len(c_id3) != 0:
            r_id = random.randint(0, len(c_id3) - 1)
            c_id = c_id3[r_id]
        if c_id == -1:
            return 'No movies matching any of these genres'
        else:
            return self.movie_db[c_id - 1].split('|')[1]






    
    


if __name__ == '__main__':
    c_model = Complex_model(MODEL_LOC)
    c_model.run_tests(TEST_DIR)
