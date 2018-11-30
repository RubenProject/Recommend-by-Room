from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os


IMG_DIR = '../img2/'
DESC_FILE = '../desc/name_genres.txt'
GENRE_COUNT = 28
FILE_COUNT = 21213
GENRE_DICT = {'Ac': 0, 'Adu': 1,
        'Adv': 2, 'An': 3,
        'B': 4, 'Co': 5,
        'Cr': 6, 'Dr': 7,
        'Do': 8, 'Fam': 9,
        'Fan': 10, 'Fi': 11,
        'Ga': 12, 'Hi': 13,
        'Ho': 14, 'Mu1': 15,
        'Mu2': 16, 'My': 17,
        'N': 18, 'Re': 19,
        'Ro': 20, 'Sc': 21,
        'Sh': 22, 'Sp': 23,
        'Ta': 24, 'Th': 25,
        'Wa': 26, 'We': 27}

fpd = open(DESC_FILE, 'r')
MOVIE_DATA = fpd.readlines()


def get_label(fname):
    y = np.zeros(GENRE_COUNT, dtype=np.float64)
    iid = int(fname.split('_')[0])
    line = MOVIE_DATA[iid-1].split('|')
    if len(line) > 1:
        genres = line[2]
    for genre in genres.split(' '):
        if genre != '' and genre != '\n':
            y[GENRE_DICT[genre]] = 1
    
    y = np.array([y])
    return y


def gen_HSV_hist(start, stop):
    print("preparing hist: ", start, " to ", stop)
    for subdir, dirs, files in os.walk(IMG_DIR):
        x = np.empty((stop - start, 512))
        y = np.empty((stop - start, GENRE_COUNT))
        for i, f in enumerate(files):
            if i < start:
                continue
            if i == stop:
                break
            img_path = IMG_DIR + f
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            H, S, V = cv.split(img)
            hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            x[i] = hist.flatten()
            y[i - start] = get_label(f)
    return x, y


def gen_data(start, stop):
    print("preparing data: ", start, " to ", stop)
#224 x 224 pixels x 3 dimensional color space
#genre_count size one-hot vector
    for subdir, dirs, files in os.walk(IMG_DIR):
        x = np.empty((stop - start, 224, 224, 3))
        y = np.empty((stop - start, GENRE_COUNT))
        for i, f in enumerate(files):
            if i < start:
                continue
            if i == stop:
                break
            img_path = IMG_DIR + f
            img = image.load_img(img_path, target_size=(224, 224))
            x[i - start] = image.img_to_array(img)
            x[i - start] = np.expand_dims(x[i - start], axis=0)
            x[i - start] = preprocess_input(x[i - start])
            y[i - start] = get_label(f)
    return x, y


def gen_BGR_hist(start, stop):
    print("preparing hist: ", start, " to ", stop)
    for subdir, dirs, files in os.walk(IMG_DIR):
        x = np.empty((stop - start, 512))
        y = np.empty((stop - start, GENRE_COUNT))
        for i, f in enumerate(files):
            if i < start:
                continue
            if i == stop:
                break
            img_path = IMG_DIR + f
            img = cv.imread(img_path)
            hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            x[i - start] = hist.flatten()
            y[i - start] = get_label(f)
    return x, y


def create_HIST_model():
    input_shape = (512,)
    hist_input = Input(input_shape)
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(GENRE_COUNT, activation='softmax'))
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
    model.summary()
    return model


def train_hist(model, epochs):
    x, y = gen_BGR_hist(0, FILE_COUNT)
    model.fit(x=x, y=y, batch_size=32, verbose=True, epochs=epochs)
    return model


def create_vgg19(pooling):
    model = VGG19(weights='imagenet', include_top=False, pooling=pooling)
    for layer in model.layers:
        layer.trainable = False
    return model


def create_resnet(pooling):
    model = ResNet50(weights='imagenet', include_top=False, pooling=pooling)
    for layer in model.layers:
        layer.trainable = False
    return model


def merge_model_2(v_model, r_model):
    input_layer = Input((224, 224, 3,))
    x = v_model(input_layer)
    y = r_model(input_layer)
    x = concatenate([x, y], axis=-1)
    x = Dense(300, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    preds = Dense(GENRE_COUNT, activation='softmax')(x)
    model = Model(input=input_layer, output=preds)
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file='../nets/mixed_2.png')
    return model


def merge_model_3(h_model, v_model, r_model):
    input_layer = Input((224, 224, 3,))
    input_layer_h = Input((512,))
    x = v_model(input_layer)
    y = r_model(input_layer)
    z = h_model(input_layer_h)
    x = concatenate([x, y, z], axis=-1)
    x = Dense(300, activation='sigmoid')(x)
    x = Dense(100, activation='sigmoid')(x)
    preds = Dense(GENRE_COUNT, activation='softmax')(x)
    model = Model(inputs=[input_layer, input_layer_h], output=preds)
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file='../nets/mixed_3.png')
    return model


def train_mixed_3(model, epochs):
    nsteps = 10
    step = int(FILE_COUNT / nsteps)
    for i in range(epochs):
        print("Epoch ", i+1, " of ", epochs)
        start = 0
        stop = int(FILE_COUNT / nsteps)
        while stop < FILE_COUNT:
            x1, y = gen_data(start, stop)
            x2, y = gen_BGR_hist(start, stop)
            hist = model.fit(x=[x1, x2], y=y, batch_size=32, verbose=False)
            print("loss: ", hist.history['loss'][0])
            start += step
            stop += step
    return model



if __name__ == '__main__':
    #hist_model = create_HIST_model()
    #hist_model = train_hist(hist_model, 10)
    #hist_model.save('../nets/hist_sig_100_30.h5')
    hist_model = load_model('../nets/hist_sig_100_30.h5')
    for layer in hist_model.layers:
        layer.trainable = False
    VGG19_model = create_vgg19('avg')
    resnet_model = create_resnet('avg')
    #mixed_model = merge_model_2(VGG19_model, resnet_model)
    mixed_model = merge_model_3(hist_model, VGG19_model, resnet_model)
    mixed_model = train_mixed_3(mixed_model, 10)
    mixed_model.save('../nets/mixed_3_300_100_3.h5')

