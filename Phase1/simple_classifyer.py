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
    valid_img_type = [".jpg"] # all images are type jpg
    img_hog = []
    # laod all image addrs into a list
    img_addrs_list = glob.glob(datapath + '/*' + valid_img_type[0])
    counter = 0
    for i in chunked_iterable(img_addrs_list,size=batch_size): # iterate over each batch
            counter +=1
            img_addr_batch = i       
            for addr in img_addr_batch:
                img = ski.imread(addr)
                hog_vector = hog(img,orientations=9,pixels_per_cell=(64,64))
                img_hog.append(hog_vector)
            print('processing batch {} of {}'.format(counter,len(img_addrs_list)//batch_size),'\n')
    img_hog = np.array(img_hog) # transform to np-array
    return img_hog,img_addrs_list

''
def normalize_by_length(feat_array,norm_order=1):
    # Auxiliary function to normalize feature vectors
    feat_array_norm = []
    for f in feat_array:
        f_norm = np.linalg.norm(f,ord=norm_order) # L1 Norm by default
        feat_array_norm.append(f/f_norm)
    feat_array_norm = np.array(feat_array_norm)   
    return feat_array_norm 


def main():

    # define variables to extract features
    path_train = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Training_Input'
    path_test = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Test_Input'
    batch_size = 20
    # Extraxt features
    img_train_hog,img_train_addrs_list = load_data_batch(path_train,batch_size)
    img_test_hog,img_test_addrs_list = load_data_batch(path_test,batch_size)
    print('length of img_hog: ' , len(img_train_hog))
    print('length of each hog_vector:' , len(img_train_hog[0]))
    
    # normalize feature vector
    norm_order = 1
    img_train_hog_norm = normalize_by_length(img_train_hog,norm_order)
    img_test_hog_norm = normalize_by_length(img_test_hog,norm_order)

    # Split training data and validation data
    


    # Train classifier

    # Validation

    # Test

if __name__ == "__main__":
    main()





        





