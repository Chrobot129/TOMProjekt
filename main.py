#%%
from preprocessing import preprocessing, size_normalization
import matplotlib.pyplot as plt
import nibabel
import os

#%%


for case_nr in range(210,210):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

#%%
preprocessing(50, 30)

# %%
print("max size: {}".format(size_normalization()))

# %%
os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\preprocessed_added_dummy") 

case_nb = nibabel.load('case{}_preprocessed_added.nii.gz'.format(3))
case = case_nb.get_fdata()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.5))
fig.suptitle('Case: {}, Slice: {}'.format(3, 20), fontsize=16)
ax.imshow(case[20,:,:]  , cmap='gray')
ax.set_title('Original')
ax.axis('off')

# %%
