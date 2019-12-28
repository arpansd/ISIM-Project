# Pahse 1 of ISIM Project
# Simple Classification Task

import matplotlib.pyplot as plt
import numpy as np
import glob
import os, os.path
import scipy as sc
import skimage.io as ski
from skimage.feature import hog
import cv2
from PIL import Image
import itertools
import pandas as pd
import ntpath

def chunked_iterable(iterable, size):
# Auxiliary function to iterate over chunks
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def load_data_batch(datapath,batch_size = 20):
# Auxiliary Function to load images batchwise from a specific path
    valid_img_type = '.jpg' # all images are type jpg
    img_hog = []
    # laod all image addrs into a list
    img_addrs_list = glob.glob(datapath + '/*' + valid_img_type)
    img_ident_list = [None] * len(img_addrs_list) 
    for i,addr in enumerate(img_addrs_list): 
        img_ident = ntpath.basename(addr)
        img_ident = img_ident[:-len(valid_img_type)]
        img_ident_list[i] = img_ident # list of img names
    counter = 0
    for i in chunked_iterable(img_addrs_list,size=batch_size): # iterate over each batch
            counter +=1
            img_addr_batch = i       
            for addr in img_addr_batch:
                img = ski.imread(addr)
                hog_vector = hog(img,orientations=9,pixels_per_cell=(64,64))
                img_hog.append(hog_vector) # TODO allocate list, because appending is inefficient!
            print('processing batch {} of {}'.format(counter,len(img_addrs_list)//batch_size),'\n')
    img_hog = np.array(img_hog) # transform to np-array
    return img_hog, img_ident_list, img_addrs_list

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

def sort_data(data,ident_data,ident_reference):
    # Auxiliary Fct to sort data acc. to a sorted ident list
    (_,dim_data) = data.shape
    n_data_sorted = len(ident_reference)-1 # length without header
    data_sorted = np.empty([n_data_sorted,dim_data])
    idx_sorted = np.empty([n_data_sorted])
    for i,ident in enumerate(ident_reference['image']):
        idx_sorted[i] = ident_data.index(ident)
        data_sorted[i] = data[idx_sorted[i]]

    return data_sorted


def main():

    # data paths and groundtruth
    path_train = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Training_Input'
    path_test = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Test_Input'
    path_train_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_train.csv'
    path_val_gt = '/Users/meko/Documents/Repos/ISIM-Project_local/data/groundtruth_val.csv'
    gt_train = pd.read_csv(path_train_gt,sep=',')
    gt_val = pd.read_csv(path_val_gt,sep=',')
    
    # set variables
    

    # Extract features 
    hog_feat_train,ident_train = load_data_batch(path_train, batch_size=20)
    hog_feat_test,ident_train = load_data_batch(path_test, batch_size=20)
    print('length of img_hog: ' , len(hog_feat_train))
    print('length of each hog_vector:' , len(hog_feat_train[0]))
    
    # normalize feature vector
    norm_order = 1
    hog_feat_train_norm = normalize_by_length(hog_feat_train,norm_order)
    hog_feat_test_norm = normalize_by_length(hog_feat_test,norm_order)

    # Split into training data and validation data according to ground truth 
    hog_feat_train_sorted = sort_data(hog_feat_train_norm,ident_train,gt_train)
    hog_feat_val_sorted = sort_data(hog_feat_train_norm,ident_train,gt_val)

    # Train classifier

    # Validation

    # Test

if __name__ == "__main__":
    main()

