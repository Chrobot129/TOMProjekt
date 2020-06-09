#%%
from starter_code.utils import load_segmentation
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import numexpr as ne
from joblib import Parallel, delayed
import multiprocessing
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
import pandas as pd


#%%

def get_data(case_nr):

    def get_case(case_nr):
        
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\preprocessed_added_dummy")

        case_nb = nibabel.load('case{}_preprocessed_added.nii.gz'.format(case_nr))
        case = case_nb.get_fdata()

        return case

    def get_segment(case_nr):
        
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\segment_added_dummy")

        segment_nb = nibabel.load('segmentation{}_added.nii.gz'.format(case_nr))
        segment = segment_nb.get_fdata()

        return segment

    X = get_case(case_nr)
    Y = get_segment(case_nr)
    return X,Y

#%%
def convert_data(start_case, end_case):
    
    def single_case_conversion(case_nr):
        X,Y = get_data(case_nr)
        size = X.shape[0] * X.shape[1] * X.shape[2]

        X_conv = np.zeros((size, 4), dtype = np.float16)

        #X_conv = X.flatten()

        for z_cord in range(X.shape[0]):
            for y_cord in range(X.shape[1]):
                for x_cord in range(X.shape[2]):

                    point_id = z_cord*y_cord*x_cord

                    X_conv[point_id, 0] = x_cord
                    X_conv[point_id, 1] = y_cord
                    X_conv[point_id, 2] = z_cord
                    X_conv[point_id, 3] = X[z_cord, x_cord, y_cord]
                    
        Y_conv = Y.flatten()
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\X_csv")
        np.savez_compressed('X{}'.format(case_nr), X_conv)

        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\Y_csv")
        np.savez_compressed('Y{}'.format(case_nr), Y_conv)

        return size

    case_nr_list = range(start_case, end_case)

    for case_nr in case_nr_list:
        single_case_conversion(case_nr)
    

#%%
convert_data(0,20)
#%%

def get_conv_data(case_nr):
    os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\X_csv")
    X = np.load('X{}.npz'.format(case_nr))
    X = X["arr_0.npy"]

    os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\Y_csv")
    Y = np.load('Y{}.npz'.format(case_nr))
    Y = Y["arr_0.npy"]

    return X,Y

# %%
clf= SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1)

cases_nr_list = range(2)

Y_test = get_conv_data(0)[1]
#%%
labels = np.unique(Y_test)
#%%
for case_nr in cases_nr_list:
    X,Y = get_conv_data(case_nr)

    clf.partial_fit(X,Y, labels)

#%%

test_case_list = range(2, 3)

for case_nr in test_case_list:
    X,Y = get_conv_data(case_nr)
    print(clf.score(X,Y ))

# %%
