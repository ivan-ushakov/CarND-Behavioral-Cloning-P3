import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Convolution2D, Flatten, Dense

from sklearn.model_selection import train_test_split

from random import shuffle

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # original image
                if batch_sample[2] == 'O':
                    images.append(cv2.imread(batch_sample[0]))
                    angles.append(batch_sample[1])
                # left-right flipped image
                if batch_sample[2] == 'F':
                    images.append(np.fliplr(cv2.imread(batch_sample[0])))
                    angles.append(-batch_sample[1])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def train(model, samples):
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, 
        nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
    
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model MSE loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('training.png')

# NVIDIA CNN
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        for i in range(3):
            angle = float(line[3])
                
            # left camera
            if i == 1:
                angle = angle + 0.2
            # right camera
            if i == 2:
                angle = angle - 0.2

            samples.append([line[i], angle, 'O'])
            samples.append([line[i], angle, 'F'])

print('number of samples: {}'.format(len(samples)))
train(model, samples)

model.save('model.h5')
