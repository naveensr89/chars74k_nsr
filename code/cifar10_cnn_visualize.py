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
import theano
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
#    ax.axis('off')

import numpy.ma as ma

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

model_fname = '../model/cifar10_CNN_model.h5'
cmap = pl.cm.binary

# the data, shuffled and split between tran and test sets
X_tr_orig = np.load('../other/X_tr_orig.npy')
X_train = X_tr_orig.reshape(X_tr_orig.shape[0], 1, 20, 20)
X_train = X_train.astype("float32")
X_train /= 255

# Model
img_channels = 1
img_rows = img_cols = 20
nb_classes = 62
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='full',
                        input_shape=(img_channels, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(32, 3, 3))
convout2 = Activation('relu')
model.add(convout2)
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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.load_weights(model_fname)

# from keras.utils.visualize_util import plot
# plot(model, to_file='../other/figures/cnn_model.png')
# exit()

convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
convout2_f = theano.function([model.get_input(train=False)], convout2.get_output(train=False))

# Convolution layer 1 weights
W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape)

pl.figure(figsize=(15, 15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cmap)
pl.savefig('../other/figures/cnn_weights.png',bbox_inches='tight', dpi=200)
pl.show()


# Visualize convolution 1 result (after activation)
C1 = convout1_f(X_train[int(sys.argv[1]):int(sys.argv[1])+1])
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.title('convout1')
nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cmap)
pl.savefig('../other/figures/cnn_convout1.png',bbox_inches='tight', dpi=200)
pl.show()

# example number 4 looks good

# Visualize convolution 2 result (after activation)
C1 = convout2_f(X_train[int(sys.argv[1]):int(sys.argv[1])+1])
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.title('convout2')
nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cmap)
pl.savefig('../other/figures/cnn_convout2.png',bbox_inches='tight', dpi=200)
pl.show()
