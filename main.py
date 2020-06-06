#%%
from preprocessing import preprocessing, size_normalization
import time

#%%


for case_nr in range(210,210):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

#%%
preprocessing(50, 30)

# %%
print("max size: {}".format(size_normalization()))

# %%
