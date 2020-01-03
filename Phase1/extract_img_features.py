import glob 
import numpy as np
import ntpath
from skimage.feature import hog
from skimage.transform import resize
import cv2
import itertools
import skimage.io as ski


def extract_img_features(datapath, batch_size = 20):
# Auxiliary Function to load images batchwise from a specific path and extract features
# Input:    - datapath
#           - batch_size
#           - feature_extractor TODO: different feature extractor?
# Output:   - feature_vect
#           - img_id_list
    
# helper functions___________________________________________________________    
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
#_____________________________________________________________________________    
    out_size = 512 # resize pixel #TODO
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
                img = resize(img,[out_size, out_size],order=1, # resize image TODO intelligent cropping here
                            mode='constant',
                            cval=0, clip=True, 
                            preserve_range=True,
                            anti_aliasing=True)    

                hog_vector = hog(img,orientations=9,pixels_per_cell=(64,64),block_norm='L2-Hys',multichannel=True)
                img_hog.append(hog_vector) # TODO allocate list, because appending is inefficient!
            print('processing batch {} of {}'.format(counter,len(img_addrs_list)//batch_size))
    img_hog = np.array(img_hog) # transform to np-array
    feature_vect = img_hog
    return feature_vect, img_id_list
