# Phase 1 of ISIM Project
# Simple Classification Task

import matplotlib.pyplot as plt
import numpy as np
import glob
import os, os.path
import scipy as sc
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score #metrics for model eval.
import skimage.io as ski
from skimage.feature import hog
from skimage.transform import resize
import cv2
import itertools
import pandas as pd
import ntpath
import time

def chunked_iterable(iterable, size):
# Auxiliary function to iterate over chunks
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl1

def load_data_batch(datapath,batch_size = 20):
# Auxiliary Function to load images batchwise from a specific path
    valid_img_type = '.jpg' # all images are type jpg
    # laod all image addrs into a list
    img_addrs_list = glob.glob(datapath + '/*' + valid_img_type)
    img_id_list = [None] * len(img_addrs_list) 
    img_hog = []
    for i,addr in enumerate(img_addrs_list): 
        img_id = ntpath.basename(addr)
        img_id = img_id[:-len(valid_img_type)]
        img_id_list[i] = img_id # list of img names
    counter = 0
    for batch in chunked_iterable(img_addrs_list,size=batch_size): # iterate over each batch
            counter += 1      
            for addr in batch:
                img = ski.imread(addr)
                img = clahe(img) # apply clahe
                img = resize(img,[512, 512]) # resize image
                hog_vector = hog(img,orientations=9,pixels_per_cell=(64,64),block_norm='L2-Hys')
                img_hog.append(hog_vector) # TODO allocate list, because appending is inefficient!
            print('processing batch {} of {}'.format(counter,len(img_addrs_list)//batch_size))
    img_hog = np.array(img_hog) # transform to np-array
    return img_hog, img_id_list

def normalize_by_length(feat_array,norm_order=1):
    # Auxiliary function to normalize feature vectors
    feat_array_norm = []
    for f in feat_array:
        f_norm = np.linalg.norm(f,ord=norm_order) # L1 Norm by default
        feat_array_norm.append(f/f_norm)
    feat_array_norm = np.array(feat_array_norm)   
    return feat_array_norm 

def train_val_split(dataset,ident_dataset,label_train,label_val):
    # Auxiliary Fct to split the data_set into train and validation data acc. to groundtruth
    (_,dim_data) = dataset.shape
    n_train = len(label_train)-1
    n_val = len(label_val)-1
    data_train = np.empty([n_train,dim_data])
    idx_train = np.empty([n_train])
    data_val = np.empty([n_val,dim_data])
    idx_val = np.empty([n_val])

    for idx,img_ident in enumerate(label_train['image']):
        idx_train[idx] = ident_dataset.index(img_ident) 
        data_train[idx] = dataset[idx_train]
    for idx,img_ident in enumerate(label_val['image']):
        idx_val[idx] = ident_dataset.index(img_ident)
        data_val[idx] = dataset[idx_val]
    
    return data_train, data_val

def sort_data(data,id_filtered,id_ref):
    # Auxiliary Fct to sort data acc. to a sorted ident list
    #_,dim_data = data.shape TODO only gives shape of axis 0
    dim_data = len(data[0])
    n_data_filtered = len(id_filtered) # length without header
    data_filtered = np.empty([n_data_filtered,dim_data])
    #idx_filtered = np.empty([n_data_filtered],dtype=int)
    print('len data_filtered:{}, shape:{}'.format(len(data_filtered),data_filtered.shape))
    print('len id_filtered:', len(id_filtered['image']))
    for i,id in enumerate(id_filtered['image']):
        #idx_filtered[i] = id_ref.index(id)
        idx = id_ref.index(id)
        data_filtered[i,:] = data[idx]

    return data_filtered

#########################################################################################################
############################ main #######################################################################
#########################################################################################################

def main():
    t = time.time() # measure execution time
    
    # data paths and groundtruth
    path_train = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Training_Input_red'
    path_test = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Test_Input_red'
    path_train_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_train_red.csv'
    path_val_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_val_red.csv'
    gt_train = pd.read_csv(path_train_gt,sep=';') # TODO red: ';'
    gt_val = pd.read_csv(path_val_gt,sep=',')
    gt_train_array = gt_train.values
    gt_train_array = gt_train_array[:,1:]
    gt_val_array = gt_val.values
    gt_val_array = gt_val_array[:,1:]
    # set variables
    scaling_switch = False # True: scaling on

    # Extract features 
    hog_feat_train,id_train = load_data_batch(path_train, batch_size=20)
    print('length of hog_train: ' , len(hog_feat_train))
    print('shape of hog_train:' , hog_feat_train.shape)
    print('length of each hog vect:' , len(hog_feat_train[0]))
    hog_feat_test,ident_train = load_data_batch(path_test, batch_size=40)
    
    elapsed = time.time() - t
    print('elapsed time = ', elapsed)

    # Split into training data and validation data according to ground truth 
    #print('id_train: ', id_train)
    #print('id_train type:', type(id_train))
    #print('id_train len:', len(id_train))

    hog_feat_train_sorted = sort_data(hog_feat_train,gt_train,id_train)
    hog_feat_val_sorted = sort_data(hog_feat_train,gt_val,id_train)
    
    # scale feature vector
    if scaling_switch == True:
        scaler = preprocessing.StandardScaler().fit(hog_feat_train_sorted) # train scaler on train data
        hog_feat_train_sorted = scaler.transform(hog_feat_train_sorted)
        hog_feat_val_sorted = scaler.transform(hog_feat_val_sorted)
        hog_feat_test = scaler.transform(hog_feat_test)
        print('Mean of scaled train data = {}, std = {}'.format(hog_feat_train_sorted.mean(axis=0),
                                                                hog_feat_val_sorted.std(axis=0)))
        '''
        norm_order = 1
        hog_feat_train_scaled = normalize_by_length(hog_feat_train,norm_order)
        hog_feat_test_scaled = normalize_by_length(hog_feat_test,norm_order)
        '''
    else:
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

    # Declare hyperparameters 
    hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

    # Find best fitting hyperparameters
    clf = GridSearchCV(pipeline,hyperparameters,cv=10)
    # Fit and tune model
    clf.fit(hog_feat_train_sorted, gt_train_array)
    
    # Test
    label_pred = clf.predict(hog_feat_val_sorted)
    print('label_pred: ', label_pred)
    r2_score_val = r2_score(gt_val_array,label_pred)
    mse_val = mean_squared_error(gt_val_array,label_pred)
    print('r2_score = {}, mse = {}'.format(r2_score_val,mse_val))
    
if __name__ == "__main__":
    main()

