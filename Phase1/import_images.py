# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:49:40 2019

@author: Jonas Röper
"""

#radditional equired moduls:
#pip install Pillow

# code snippet just for getting startet and test feature extraction methods
# avoid saving all images for working with the full dataset - features only ;)

import numpy as np
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt
import cv2
#%matplotlib qt

def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]
	# return our set of features
	return features

#def color_constancy(img, power=6, gamma=None)
def color_constancy(img,pw,gamma):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype
    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
            img = cv2.LUT(img, look_up_table)
            
    img = img.astype('float32')
    img_power = np.power(img, pw)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/pw)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    return img.astype(img_dtype)

imgs=[]
features=[]
path = os.path.dirname(os.path.realpath(__file__))+ "/img"
#valid_images = [".jpg",".gif",".png",".tga"]

valid_images = [".jpg"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    temp=[]
    img=Image.open(os.path.join(path,f))
    imgs.append(img)
    img_gray=img.convert('L') # translate color image to grayscale
    img_grayscaled=color_constancy(np.array(img),6,None)
    
    temp.append(extract_color_stats(img))
    temp.append(img_gray)
    temp.append(img_grayscaled)
    features.append(temp)

#pick and plot random image
NUM_PLOTS=5
n=np.random.choice(range(len(imgs)), NUM_PLOTS, replace=False)
plt.figure(0,figsize=[12,4*NUM_PLOTS])
for i in range(0,NUM_PLOTS):
    n_im=n[i]
    
    plt.subplot(NUM_PLOTS,3,3*i+1)
    plt.imshow(imgs[n_im])
    plt.title('Original image #'+str(n_im))
    plt.subplot(NUM_PLOTS,3,3*i+2)
    plt.imshow(features[n_im][1], cmap='gray', vmin=0, vmax=255)
    plt.title('Grayscale image #'+str(n_im))
    plt.subplot(NUM_PLOTS,3,3*i+3)
    plt.imshow(features[n_im][2])
    plt.title('"Shades of grey"  image #'+str(n_im))
plt.show()
