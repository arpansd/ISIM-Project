import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cv2 as cv
import math 


'''
Function to calculate the Hu moments of an image

INPUT:
image (rgb or grayscale)

OUTPUT:
feature vector containing the Hu moments
'''

def calc_hu_moments(img):

    # calculate moments
    moments = cv.moments(img)
    # calculate hu moments
    hu_moments = cv.HuMoments(moments)
    # Do log transform for comparable scale
    for i in range(0,7):
        hu_moments[i] = -1* math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

    
