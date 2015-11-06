__author__ = 'naveen'
import os
import random
import numpy as np
import pandas as pd
import cv2

def tile(wd,ht,rows,cols,input,y_tr, labels_string, color=0):
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

    if folder_passed == 1 and color == 1:
        out = np.zeros((ht,wd,3),dtype=np.uint8)
        clr_idx = cv2.IMREAD_COLOR
    else:
        out = np.zeros((ht,wd),dtype=np.uint8)
        clr_idx = cv2.IMREAD_GRAYSCALE

    iwd = wd / cols
    iht = ht / rows

    # overlay Text font
    font = cv2.FONT_HERSHEY_TRIPLEX

    for row in range(rows):
        for col in range(cols):
            idx = random.randint(0,numfiles)
            label = labels_string[y_tr[idx]]
            if folder_passed == 1:
                fname = os.path.splitext(files[idx])[0]
                # print idx,files[idx]
                # print fname,label
                img = cv2.imread(folder+'/'+files[idx],clr_idx)
                img = cv2.resize(img,(iht,iwd),interpolation = cv2.INTER_CUBIC)
            else:
                img = X[idx,]
                d = np.sqrt(X.shape[1])
                img = img.reshape((d,d)) # Assuming square image
                img = cv2.resize(img,(iht,iwd),interpolation = cv2.INTER_LINEAR)

            cv2.putText(img,label,(0,3*iht/4), font, 1,(255,255,255),1,cv2.LINE_AA)

            if folder_passed == 1 and color == 1:
                out[row*iht:(row+1)*iht,col*iwd:(col+1)*iwd,] = img
            else:
                out[row*iht:(row+1)*iht,col*iwd:(col+1)*iwd] = img

    cv2.imshow('Collage',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
