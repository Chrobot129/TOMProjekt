#%%
from starter_code.utils import load_segmentation
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import numexpr as ne
from joblib import Parallel, delayed
import multiprocessing
import os
from thundersvm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import dask.array as da 
#from sklearn.svm import SVC

#Some case files are corrupted. For now only 20 cases.

#%%

def get_dataset(num_of_cases):


    X = da.zeros((num_of_cases,834,512,512), dtype=np.float16)
    Y = da.zeros((num_of_cases,834,512,512), dtype=np.float16)

    def get_case(case_nr, target_array):
        
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\preprocessed")

        case_nb = nibabel.load('case{}_preprocessed.nii.gz'.format(case_nr))
        case = da.from_array(case_nb.get_fdata())
        size = case.shape[0]

        X[case_nr,:size,:,:] += case

        return target_array

    case_nr_list = range(num_of_cases)

    X_full = map(get_case, case_nr_list)
    Y_full = map(get_case, case_nr_list)   

    return X_full,Y_full


#%%
X,Y = get_dataset(20)
#%%


X = X.compute()
Y = Y.compute()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle = True)
#%%
svc = SVC(kernel='poly', degree=6, C = 0.8, coef0=15)
#%%

scores = cross_val_score(svc, X_train, y_train, cv=5)

print('Accuracy: ',scores)
print('Accuracy (mean): ', scores.mean())
print('Accuracy (std): ', scores.std())

# %%
