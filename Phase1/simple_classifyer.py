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
# helper function to itarate over chunks
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
def normalize(feat_array):
    # Auxiliary function to normalize feature vectors
    f_norm = []
    return f_norm 


def main():

    # define variables to extract features
    datapath = '/Users/meko/Documents/Repos/ISIM-Project_local/data/ISIC_2019_Training_Input'
    batch_size = 20
    img_hog,img_addrs_list = load_data_batch(datapath,batch_size)
    print('length of img_hog: ' , len(img_hog))
    print('length of each hog_vector:' , len(img_hog[0]))
    
    # normalize feature vector

    # Define training data and validation data

    # Train classifier

    # Validation

    # Test







        





