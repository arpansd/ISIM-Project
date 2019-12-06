import pandas as pd
import numpy as np
import glob

filepath = '/Users/meko/Documents/Repos/ISIM-Project/data/' 

tr_input_path = filepath + 'ISIC_2019_Training_Input/*.jpg' # images of skin lesion
tr_meta_path = filepath + 'ISIC_2019_Training_Metadata.csv' # metadata (age,sex,general anatomic site, common lesion identifier)
tr_ground_path = filepath + 'ISIC_2019_Training_GroundTruth.csv' # gold standard lesion diagnosis
test_input_path = filepath + 'ISIC_2019_Test_Input/*.jpg' # images of skin lesion
test_meta_path = filepath + 'ISIC_2019_Test_Metadata.csv' # metadata(age, sex, general anatomic site)

# Read csv data as numpy array
tr_meta = pd.read_csv(tr_meta_path, sep=',')
tr_ground = pd.read_csv(tr_ground_path,sep=',')
te_meta = pd.read_csv(test_meta_path,sep=',')

# save image paths as list
tr_addrs = glob.glob(tr_input_path) 
te_addrs = glob.glob(test_input_path)

_,tr_label_columns = tr_ground.shape 

# print description of train data
print(tr_ground.head())
print(tr_ground.describe())
print(tr_meta.head())




print(tr_addrs[0])