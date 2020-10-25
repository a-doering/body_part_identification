import time
import numpy as np
from medio import get_paths
from medio import read_image
from util import patches

def prepare_data(patch_size, overlap,  *paths):
    '''Prepare images, labels, patch_pos dictionaries with keys for data generator'''    
    path_images, path_labels = get_paths.get_paths(*paths, b_verbose = False)
    assert len(path_images) == len(path_labels), "Number of label and image paths not matching!"
    n_patients = len(path_images)
    
    # Read image shape from .mat files
    image_shapes = np.zeros([n_patients, 3])
    time0 = time.time()
    for idx in range(n_patients):
       image = read_image.read_mat_image(path_images[idx])
       image_shapes[idx] = image.shape
       print(image.shape)
    print('time to get_shapes: ', time.time()-time0)
    
    # Create list containing the position of all possible patch positions for all images
    print('computing patch indices')
    patch_pos = []
    overlap = overlap
    time1 = time.time()
    for idx in range(n_patients):
        patch_pos.append(patches.compute_patch_indices(image_shapes[idx], 
                                                  patch_size, 
                                                  overlap = overlap))
    print('time to compute_patch_indices: ', time.time()-time1)
    
    # Create list with all labels
    labels = []
    for idx in range(n_patients):
        labels.append(read_image.read_mat_labels(path_labels[idx]))
    
    # Create dictionaries
    dict_images = {i : path_images[i] for i in range(n_patients)}
    dict_labels = {i : labels[i] for i in range(n_patients)}
    dict_patch_pos = {i : patch_pos[i] for i in range(n_patients)}
    keys = np.arange(0,n_patients, 1)
    
    return n_patients, dict_images, dict_labels, dict_patch_pos, keys

if __name__ == '__main__':
    path_dir_1_5T = '/home/s1283/no_backup/s1283/data/1_5T'
    path_dir_3T = '/home/s1283/no_backup/s1283/data/3T'
    n_patients, dict_images, dict_labels, dict_patch_pos, keys = prepare_data([1,40,16], 0.6, path_dir_1_5T, path_dir_3T)