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
from extract_img_features import extract_img_features
import cv2
import itertools
import pandas as pd
import time


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
    # set parameters
    scaling_switch = False # True: scaling on
    t = time.time() # measure execution time
    
    # data paths
    path_train = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Training_Input_red'
    path_test = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Test_Input_red'
    path_train_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_train_red.csv'
    path_val_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_val_red.csv'

    # Import ground truth
    gt_train = pd.read_csv(path_train_gt,sep=';') # TODO red: ';'
    gt_val = pd.read_csv(path_val_gt,sep=',')
    gt_train_array = gt_train.values
    gt_train_array = gt_train_array[:,1:]
    gt_val_array = gt_val.values
    gt_val_array = gt_val_array[:,1:]
    
    # Load images and extract features
    hog_feat_train,id_train = extract_img_features(path_train, batch_size=20)
    print('length of hog_train: ' , len(hog_feat_train))
    print('shape of hog_train:' , hog_feat_train.shape)
    print('length of each hog vect:' , len(hog_feat_train[0]))
    hog_feat_test,id_test = extract_img_features(path_test, batch_size=40)
    
    elapsed = time.time() - t
    print('elapsed time = ', elapsed)

    # Split into training data and validation data according to ground truth 
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

