#%%
from preprocessing import preprocessing, size_normalization
import matplotlib.pyplot as plt
import nibabel
import os

#%%


for case_nr in range(0,20):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

# %%
size_normalization()
