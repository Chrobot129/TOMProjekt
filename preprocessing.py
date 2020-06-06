def size_normalization():

    from starter_code.utils import load_volume
    import numpy as np
    import nibabel
    import numexpr as ne
    from joblib import Parallel, delayed
    import multiprocessing
    import os

    def get_size(case_nr):
        case = load_volume(case_nr)
        case_size = case.get_fdata().shape[0]

        return case_size

    case_nr_list = range(210)
    num_cores = multiprocessing.cpu_count()
    size_list = Parallel(n_jobs=num_cores)(delayed(get_size)(case_nr) for case_nr in case_nr_list)
    size_arr = np.array(size_list)

    max_size = np.amax(size_arr)

    return max_size


def preprocessing(case_nr, slice_number_to_print):

    from starter_code.utils import load_volume
    import numpy as np
    import nibabel
    import matplotlib.pyplot as plt
    from skimage import filters
    from skimage.morphology import disk, square
    from skimage import exposure
    from skimage.morphology import disk, closing, opening, remove_small_holes
    from skimage.color import label2rgb, rgb2gray
    import numexpr as ne
    from joblib import Parallel, delayed
    import multiprocessing
    import os

    def normalize(image):
        minimum = np.min(image)
        maximum = np.max(image)
        result = (image - minimum)/(maximum - minimum)
        return result

    volume = load_volume(case_nr)
    #data_seg = segment.get_fdata()
    data = volume.get_fdata()
    data_preprocessed_vol = np.zeros(data.shape)
    data_masks = np.zeros(data.shape)

    slice_nr_list = list(range(data.shape[0])) #List to iterate by

    #Definition of preprocessing per one slice
    def slice_pre(data_slice): 

        data_normalized = normalize(data_slice)
        data_oryg = data_normalized.copy()
        data_hist = exposure.equalize_adapthist(data_normalized, clip_limit = 0.1)

        data_median = filters.median(data_hist, selem = disk(3))
        data_median = exposure.adjust_gamma(data_median, gamma = 10)

        thresholds = filters.threshold_multiotsu(data_median)
        regions = np.digitize(data_normalized, bins=thresholds)

        data_normalized[regions != 2] = 0

        data_pre = exposure.equalize_hist(data_normalized)

        thresholds = filters.threshold_multiotsu(data_pre)

        regions = np.digitize(data_pre, bins=thresholds)
        data_mask = label2rgb(regions)
        data_mask = rgb2gray(data_mask)
        data_mask[data_mask != np.max(data_mask)] = 0
        data_mask[data_mask == np.max(data_mask)] = 1
        data_mask = closing(data_mask, selem = disk(3)).astype(int)
        data_mask = remove_small_holes(data_mask, area_threshold=300)

        data_eq = exposure.equalize_adapthist(data_oryg, clip_limit = 0.15)
        data_preprocessed = data_eq*data_mask
        data_preprocessed = opening(data_preprocessed, selem = disk(3))

        return data_preprocessed, data_mask

    num_cores = multiprocessing.cpu_count()
    data_list = Parallel(n_jobs=num_cores)(delayed(slice_pre)(data[slice_nr,:,:]) for slice_nr in slice_nr_list)
    data_arr = np.array(data_list)
    data_preprocessed_vol = data_arr[:,0,:,:]
    data_masks = data_arr[:,1,:,:]

    path = os.getcwd()
    new_path = os.path.join(path, "preprocessed")
    os.chdir(new_path)

    img_pre = nibabel.Nifti1Image(data_preprocessed_vol, volume.affine)
    nibabel.save(img_pre,'case{}_preprocessed.nii.gz'.format(case_nr))
    #Saving to file
    os.chdir(path)

    #This part prints slice if user specifies slice number
    if slice_number_to_print != -1:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
        fig.suptitle('Case: {}, Slice: {}'.format(case_nr, slice_number_to_print), fontsize=16)
        ax[0].imshow(data[slice_number_to_print,:,:]  , cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(data_masks[slice_number_to_print,:,:], cmap='gray')
        ax[1].set_title('mask')
        ax[1].axis('off')

        ax[2].imshow(data_preprocessed_vol[slice_number_to_print,:,:], cmap='gray')
        ax[2].set_title('preprocessed')
        ax[2].axis('off')

        plt.subplots_adjust()

        plt.show()