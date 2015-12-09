
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from utils import tile, get_HOG_image
from skimage.feature import hog
from feature_expansion import feature_threshold, feature_sobel

X_tr_orig = np.load('../other/X_tr_orig.npy')
X_te_orig = np.load('../other/X_te_orig.npy')
y_tr_orig = np.load('../other/y_tr_orig.npy')
labels_string = np.load('../other/labels_string.npy')
sorted_files_tr_orig = np.load('../other/sorted_files_tr_orig.npy')
sorted_files_te_orig = np.load('../other/sorted_files_te_orig.npy')

out = tile(300,600,3,10,'../data/train',y_tr_orig, labels_string, 1, 42,1)
cv2.imwrite('../other/figures/visual_small.png',out)

img = cv2.imread('../data/train/1.Bmp',0)
h,h_img = hog(img, orientations=8, pixels_per_cell=(15, 15),
            cells_per_block=(2, 2), visualise=True, normalise=True)
a = h_img.min()
b = h_img.max()
h_img = (h_img - a)/(b-a) * 255
h_img_1 = h_img.astype('uint8')

cv2.imwrite('../other/figures/hog_full.png',h_img_1)

img = cv2.resize(img,(img.shape[1]/2,img.shape[0]/2),interpolation = cv2.INTER_CUBIC)

h,h_img = hog(img, orientations=8, pixels_per_cell=(15, 15),
            cells_per_block=(2, 2), visualise=True, normalise=True)
a = h_img.min()
b = h_img.max()
h_img = (h_img - a)/(b-a) * 255
h_img_2 = h_img.astype('uint8')

cv2.imwrite('../other/figures/hog_half.png',h_img_2)

X_hog = get_HOG_image(X_tr_orig)
out = tile(100,600,1,10,X_hog,y_tr_orig, labels_string, 0, 42)
cv2.imwrite('../other/figures/hog_visual.png',out)

out = tile(100,600,1,10,X_tr_orig,y_tr_orig, labels_string, 0, 42)
cv2.imwrite('../other/figures/visual_1.png',out)

X_th = feature_threshold(X_tr_orig)
X_s = feature_sobel(X_tr_orig)

out = tile(100,600,1,10,X_th,y_tr_orig, labels_string, 0, 42)
cv2.imwrite('../other/figures/visual_th.png',out)

out = tile(100,600,1,10,X_s,y_tr_orig, labels_string, 0, 42)
cv2.imwrite('../other/figures/visual_sobel.png',out)


# Data augmentation
img = cv2.imread('../other/figures/1.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,0],[0,1,30]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('../other/figures/data_aug_t1.png',dst)

M = np.float32([[1,0,30],[0,1,0]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('../other/figures/data_aug_t2.png',dst)

M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('../other/figures/data_aug_r1.png',dst)

M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('../other/figures/data_aug_r2.png',dst)

M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
dst = cv2.warpAffine(img,M,(cols,rows))
M = np.float32([[1,0,15],[0,1,0]])
dst = cv2.warpAffine(dst,M,(cols,rows))
cv2.imwrite('../other/figures/data_aug_a.png',dst)
