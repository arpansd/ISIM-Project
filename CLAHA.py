import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.filters import threshold_otsu

"""Applying CLAHE to resolve uneven illumination"""


def Clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl1
