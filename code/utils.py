__author__ = 'naveen'
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from skimage.feature import hog

def get_HOG_image(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    # X_add = np.zeros((N,32))
    X_hog = np.zeros((N,400))
    # Setup HOG descriptor
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        h,h_img = hog(img, orientations=8, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 2), visualise=True, normalise=True)
        h_img = h_img.flatten()
        a = h_img.min()
        b = h_img.max()
        h_img = (h_img - a)/(b-a) * 255
        h_img = h_img.astype('uint8')
        X_hog[i,] = h_img
    return X_hog

def tile(ht,wd,rows,cols,input,y_tr, labels_string, overlay_label, random_state, color=0):
    if(isinstance(input,str)):
        folder = input
        files = [f for f in os.listdir(folder) if f.endswith(".Bmp")]
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        numfiles = files.__len__()
        folder_passed = 1
    elif(type(input).__module__ == np.__name__):
        X = input
        numfiles = X.shape[0]
        folder_passed = 0
    else:
        print "Pass either folder or Input matrix X in input"
        return

    iwd = wd / cols
    iht = ht / rows
    gap = 5

    if folder_passed == 1 and color == 1:
        out = np.ones((ht + (rows+1)*gap,wd +(cols+1)*gap ,3),dtype=np.uint8)*255
        clr_idx = cv2.IMREAD_COLOR
    else:
        out = np.ones((ht + (rows+1)*gap,wd +(cols+1)*gap),dtype=np.uint8)*255
        clr_idx = cv2.IMREAD_GRAYSCALE

    # overlay Text font
    font = cv2.FONT_HERSHEY_TRIPLEX

    np.random.seed(random_state)

    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(0,numfiles)
            label = labels_string[y_tr[idx]]
            if folder_passed == 1:
                fname = os.path.splitext(files[idx])[0]
                # print idx,files[idx]
                # print fname,label
                img = cv2.imread(folder+'/'+files[idx],clr_idx)
                img = cv2.resize(img,(iwd,iht),interpolation = cv2.INTER_CUBIC)
            else:
                img = X[idx,]
                d = np.sqrt(X.shape[1])
                img = img.reshape((d,d)) # Assuming square image
                img = cv2.resize(img,(iwd,iht),interpolation = cv2.INTER_LINEAR)

            if overlay_label == 1:
                cv2.putText(img,label,(0,3*iht/4), font, 1,(255,255,255),1,cv2.LINE_AA)

            if folder_passed == 1 and color == 1:
                out[row*iht+(row+1)*gap:(row+1)*iht+(row+1)*gap,col*iwd+(col+1)*gap:(col+1)*iwd+(col+1)*gap,] = img
            else:
                out[row*iht+(row+1)*gap:(row+1)*iht+(row+1)*gap,col*iwd+(col+1)*gap:(col+1)*iwd+(col+1)*gap] = img

    plt.imshow(out)
    plt.axis('off')
    plt.show()
    # cv2.imshow('Collage',out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return out

def read_X(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".Bmp")]
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    f = sorted_files[0]
    img = cv2.imread(folder+'/'+f,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    img = img[:,:,1]
    wd = img.shape[0]
    ht = img.shape[1]
    N = files.__len__()
    print wd,ht,N
    X = np.zeros((N,wd*ht))
    for i in range(N):
        f = sorted_files[i]
        img = cv2.imread(folder+'/'+f,0)
        X[i,] = img.flatten()

    return X, sorted_files

def read_X_full_res(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".Bmp")]
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    N = files.__len__()
    X = []
    for i in range(N):
        f = sorted_files[i]
        img = cv2.imread(folder+'/'+f,cv2.IMREAD_GRAYSCALE)
        X.append(img)

    return X, sorted_files

def read_y(csv_file):
    # Reading Labels
    y_tr_df = pd.read_csv(csv_file,index_col = 'ID')
    y_tr_string = y_tr_df['Class'].as_matrix()

    # Converting Labels from String to Int
    labels_string = np.unique(y_tr_string)
    labels_int = range(labels_string.shape[0])
    labels_dict = dict(zip(labels_string,labels_int))

    y_tr = [labels_dict[y] for y in y_tr_string]
    return y_tr,labels_string

# Format is in Kaggle submission format
# Save test results to csv file
def save_out(y_te,labels_string, sorted_files, file_name):
    y_te_string = [labels_string[y] for y in y_te]
    ids = [f.split('.')[0] for f in sorted_files]
    y_df = pd.DataFrame({'Class': y_te_string, 'ID':ids})
    y_df.set_index('ID')
    y_df.to_csv(file_name,index=False, columns=['ID','Class'])
    return

def print_accuracy(y1,y2, string):
    if y1.shape != y2.shape:
        return "Dimensions mismatch"

    N = y1.shape[0]

    print string+" Classification Accuracy = " + str(100 - (sum(y1 != y2)*100.0/y1.shape[0])) + " %"

def load(load_from_folder, display_collage, submission):

    if load_from_folder:
        ###### Read data and save to file ####
        X_tr_orig, sorted_files_tr_orig = read_X('../data/trainResized')
        X_te_orig, sorted_files_te_orig = read_X('../data/testResized')
        y_tr_orig,labels_string = read_y('../data/trainLabels.csv')

        np.save('../other/X_tr_orig',X_tr_orig)
        np.save('../other/X_te_orig',X_te_orig)
        np.save('../other/y_tr_orig',y_tr_orig)
        np.save('../other/labels_string',labels_string)
        np.save('../other/sorted_files_tr_orig',sorted_files_tr_orig)
        np.save('../other/sorted_files_te_orig',sorted_files_te_orig)

    else:
        ###### Read data from saved file ####
        X_tr_orig = np.load('../other/X_tr_orig.npy')
        X_te_orig = np.load('../other/X_te_orig.npy')
        y_tr_orig = np.load('../other/y_tr_orig.npy')
        labels_string = np.load('../other/labels_string.npy')
        sorted_files_tr_orig = np.load('../other/sorted_files_tr_orig.npy')
        sorted_files_te_orig = np.load('../other/sorted_files_te_orig.npy')

    if display_collage:
        ##### display HOG ####
        X_hog = get_HOG_image(X_tr_orig)
        tile(600,600,10,10,X_hog,y_tr_orig,labels_string, 1, 42)
        ####################### Display Random images from Training set as collage  ######################
        # tile(600,600,10,10,X_tr_orig,y_tr_orig, labels_string,1, 42)
        # tile(600,600,10,10,'data/train',y_tr_orig, labels_string, 1, 42, 1)

    if submission != 1:
        ###### Split train and test #########
        X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te = \
            train_test_split(X_tr_orig, y_tr_orig,sorted_files_tr_orig, test_size=0.20, random_state=42)
    else:
        X_tr, X_te, y_tr, sorted_files_tr, sorted_files_te = \
            X_tr_orig, X_te_orig, y_tr_orig, sorted_files_tr_orig, sorted_files_te_orig

    return X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te
