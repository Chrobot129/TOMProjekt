def preprocessing(case_nr, slice_number_to_print):

    from starter_code.utils import load_case
    import numpy as np
    import nibabel
    import matplotlib.pyplot as plt
    from skimage import filters
    from skimage.morphology import disk, square
    from skimage import exposure
    from skimage.morphology import disk, closing, opening, remove_small_holes
    from skimage.color import label2rgb, rgb2gray
    import multiprocessing as mp
    import numexpr as ne

    def normalize(image):
        minimum = np.min(image)
        maximum = np.max(image)
        result = (image - minimum)/(maximum - minimum)
        return result

    volume, segment = load_case(case_nr)
    data_seg = segment.get_fdata()
    data = volume.get_fdata()
    data_preprocessed_vol = np.zeros(data.shape)
    data_masks = np.zeros(data.shape)

    slice_nr_list = list(range(data.shape[0]))

    def slice_pre(slice_nr):

        data_normalized = normalize(data[slice_nr,:,:])
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

        data_masks[slice_nr,:,:] = data_mask
        data_preprocessed_vol[slice_nr,:,:] = data_preprocessed

        #return data_preprocessed

    for slice_nr in slice_nr_list:
        slice_pre(slice_nr)

    #map(slice_pre,slice_nr_list)   

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

    return data_preprocessed_vol , data_seg, data_masks
