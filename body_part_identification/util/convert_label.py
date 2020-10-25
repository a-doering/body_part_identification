import numpy as np

def make_class_label(label, patch_pos, patch_size):
    '''
    Create one hot encoded class label for each patch
    1 for the label that the patch is closest to
    '''
    one_hot_label = np.zeros([patch_pos.shape[0], label.size], dtype = 'int32')
    for idx in range(patch_pos.shape[0]):
        one_hot_label[idx][np.argmin(np.abs(label-patch_pos[idx][2]-patch_size[2]//2))] = 1
        
    return one_hot_label

def make_reg_label(label, patch_pos, patch_size):
    '''
    Return absolute distance from the middle of the patch
    to the closest threshhold between labels
    '''
    reg_label = np.zeros([patch_pos.shape[0],1])
    
    threshold = np.zeros(label.size - 1)
    for idx in range(label.size-1):
        threshold[idx] = (label[idx] + label[idx+1])/2

    for idx in range(patch_pos.shape[0]):
        reg_label[idx] = np.min(np.abs(threshold - patch_pos[idx][2] - patch_size[2]/2))
    
    return reg_label