import pandas as pd
import numpy as np
import glob

filepath = '/Users/meko/Documents/Repos/ISIM-Project_local/data/' # add your directory here

tr_input_path = filepath + 'ISIC_2019_Training_Input/*.jpg' # images 
tr_meta_path = filepath + 'ISIC_2019_Training_Metadata.csv' # metadata 
tr_ground_path = filepath + 'groundtruth_train.csv' # groundtruth
val_ground_path = filepath + 'groundtruth_val.csv' # validation groundtruth
test_input_path = filepath + 'ISIC_2019_Test_Input/*.jpg' # test images
test_meta_path = filepath + 'ISIC_2019_Test_Metadata.csv' # test metadata

# Read csv data as numpy array
tr_meta = pd.read_csv(tr_meta_path, sep=',')
tr_ground = pd.read_csv(tr_ground_path,sep=',')
val_ground = pd.read_csv(val_ground_path,sep=',')
te_meta = pd.read_csv(test_meta_path,sep=',')

# save image paths as list
tr_val_addrs = glob.glob(tr_input_path) 
te_addrs = glob.glob(test_input_path)

_,tr_label_columns = tr_ground.shape 

# print description of train data
print(tr_ground.describe())

# Data augmentation
