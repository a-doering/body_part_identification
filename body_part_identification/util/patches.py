import numpy as np
import random
from medio import read_image


def extract_patches(image_path, patch_pos, n_patches, n_channels, patch_size = [1,40,16]):
    '''
    Extract n_patches from image and return them in numpy array
    assuming 2D patch with patch_size[0]==1
    '''
    image = read_image.read_mat_image(image_path)
    patches = np.zeros((n_patches, n_channels, patch_size[1], patch_size[2]),dtype='float32')
    for i,pos in enumerate(patch_pos,0):
        patches[i] = image[pos[0]:(pos[0]+patch_size[0]),pos[1]:(pos[1]+patch_size[1]), pos[2]:(pos[2]+patch_size[2])]

    return patches


def compute_patch_indices(image_shape, patch_size, overlap = 0.6, start = [0, 0, 0], order = True):
    '''
    Compute all possible patch indices, randomizes around their location
    '''
    # Get stop index to not exceed boundaries
    stop = [(i-j) for i, j  in zip(image_shape, patch_size)]
    # Get step in each dimension and rounds it up to ensure iterability
    step = [np.ceil(patch_size[i]*(1-overlap)) for i in range(len(patch_size))]

    index_list = get_set_of_patch_indices(start, stop, step, order)
    return get_random_indices (image_shape, patch_size, index_list)

def get_set_of_patch_indices(start, stop, step, order = False):
    '''
    Creats list with all patch indices, last area to be indexed will be truncated
    If order == True, the patches near the upper boundaries will not be taken
    If order == False, the patches near the lower boundaries will not be taken
    '''
    if order:
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                          start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int32)
    else:
        return np.asarray(np.mgrid[stop[0]:start[0]:-step[0], stop[1]:start[1]:-step[1],
                           stop[2]:start[2]:-step[2]].reshape(3, -1).T, dtype=np.int32)


def get_random_indices (image_shape, patch_size, index_list):
    '''
    Randomized the indices around the index values within bounds
    '''
    index0bound = image_shape[0] - patch_size[0]
    index1bound = image_shape[1] - patch_size[1]
    index2bound = image_shape[2] - patch_size[2]

    for index in index_list:
        newIndex0 = index[0] + random.randint(-10, 10)
        newIndex1 = index[1] + random.randint(-10, 10)
        newIndex2 = index[2] + random.randint(-10, 10)
        
        # Check that patch is within boundaries
        index[0] = newIndex0 if (newIndex0 <= index0bound and newIndex0 >= 0) else index[0]
        index[1] = newIndex1 if (newIndex1 <= index1bound and newIndex1 >= 0) else index[1]
        index[2] = newIndex2 if (newIndex2 <= index2bound and newIndex2 >= 0) else index[2]

    return index_list

def draw_num_patches(index_lists, num_patches):
    '''
    Draws num_patches randomly out of the number of possible patches
    '''
    rand_idx = random.sample(range(0, len(index_lists)), num_patches)
    drawn_indices = np.zeros([num_patches, 3], dtype=np.int32)

    for i in range(num_patches):
        list_idx = rand_idx[i]
        drawn_indices[i] = index_lists[list_idx]
    return drawn_indices