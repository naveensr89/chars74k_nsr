from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
from utils import save_out

batch_size = 32
nb_classes = 62
nb_epoch = 1
data_augmentation = True

submission_fname = '../submission/testLabels_cifar10_CNN_data_aug_epoch_100_zca.csv'
model_fname = '../model/cifar10_cnn_data_aug_epoch_100.h5'
zca = True

# input image dimensions
img_rows, img_cols = 20, 20
# the CIFAR10 images are RGB
img_channels = 1

# the data, shuffled and split between tran and test sets
X_tr_orig = np.load('../other/X_tr_orig.npy')
y_tr_orig = np.load('../other/y_tr_orig.npy')
X_tr_orig = X_tr_orig.reshape(X_tr_orig.shape[0], 1, 20, 20)

X_train, X_test, y_train, y_test = \
    train_test_split(X_tr_orig, y_tr_orig, test_size=0.05, random_state=42)

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='full',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.load_weights(model_fname)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# Prediction
labels_string = np.load('../other/labels_string.npy')
sorted_files_tr = np.load('../other/sorted_files_tr_orig.npy')
sorted_files_te = np.load('../other/sorted_files_te_orig.npy')

X_te_orig = np.load('../other/X_te_orig.npy')
X_te_orig = X_te_orig.reshape(X_te_orig.shape[0], 1, 20, 20)
X_te_orig = X_te_orig.astype("float32")
X_te_orig /= 255

if not data_augmentation:
    print("Not using data augmentation or normalization")
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, accuracy=True)
    print('Test accuracy:', acc)
    y_te = model.predict_classes(X_te_orig)
    save_out(y_te,labels_string,sorted_files_te,submission_fname)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=zca,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")

        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            score, acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("train accuracy", acc)])

        print("Testing...")

        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score, acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("test accuracy", acc)])

        # test time!
        for X_batch, Y_batch in datagen.flow(X_te_orig, np.ones((1,X_te_orig.shape[0])), batch_size = X_te_orig.shape[0]):
            y_te = model.predict_classes(X_batch)

        save_out(y_te,labels_string,sorted_files_te,submission_fname)
