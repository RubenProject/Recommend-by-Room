from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
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

def create(pooling, dense_1_size, dense_2_size):
    base_model = VGG19(weights='imagenet', include_top=False, pooling=pooling)
    x = base_model.output
    x = Dense(dense_1_size, activation='relu')(x)
    x = Dense(dense_2_size, activation='relu')(x)
    preds = Dense(GENRE_COUNT, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()
    return model


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


def train_upper(model, epochs):
    print("Training upper layers of network")
#free layers that should not be trained
    for layer in model.layers:
        layer.trainable = True 
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    nsteps = 10
    step = int(FILE_COUNT / nsteps)
    for i in range(epochs):
        print("Epoch ", i, " of ", epochs)
        start = 0
        stop = int(FILE_COUNT / nsteps)
        while stop < FILE_COUNT:
            x, y = gen_data(start, stop)
            hist = model.fit(x=x, y=y, batch_size=32, verbose=False)
            print("loss: ", hist.history['loss'][0])
            start += step
            stop += step
    return model


def train_lower(model, epochs):
    print("Training lower layers of network")
#free layers that should not be trained
    for layer in model.layers:
        layer.trainable = True 
    for layer in model.layers[-3:]:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    nsteps = 10
    step = int(FILE_COUNT / nsteps)
    for i in range(epochs):
        print("Epoch ", i, " of ", epochs)
        start = 0
        stop = int(FILE_COUNT / nsteps)
        while stop <= FILE_COUNT:
            x, y = gen_data(start, stop)
            hist = model.fit(x=x, y=y, batch_size=32, verbose=False)
            print("loss: ", hist.history['loss'][0])
            start += step
            stop += step
    return model



def train(model, u1_epochs, l1_epochs, u2_epochs):
    model = train_upper(model, u1_epochs)
    model = train_lower(model, l1_epochs)
    model = train_upper(model, u2_epochs)
    return model



if __name__ == '__main__':
    model = create('max', 300, 100)
    model = train(model, 10, 50, 10)
    model.save('../nets/VGG19_max_300_100.h5')
    model = create('avg', 300, 100)
    model = train(model, 10, 50, 10)
    model.save('../nets/VGG19_avg_300_100.h5')

