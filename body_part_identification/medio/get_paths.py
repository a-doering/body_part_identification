import os
import logging
from glob import glob


def get_paths(*args, b_verbose=False, b_skip_existing=False):
    '''Return list of all image paths and list of all label paths'''
    list_image = []
    list_labels = []
    
    # Iterate through given directories and subject folders therein
    for arg in args:
        subjects = glob(os.path.join(arg, '*/'))

        for idx in range(len(subjects)):
            
            if b_verbose:
                print('subject path: ', subjects[idx])
                
            filename_labels =[filename for filename in os.listdir(subjects[idx]) if filename.startswith("labels")][0]
            path_image = os.path.join(subjects[idx], 'rework.mat')
            path_labels = os.path.join(subjects[idx], filename_labels)
            
            # Check if both label and img mat data are in the dir
            if not (os.path.isfile(path_image) and os.path.isfile(path_labels)):
                print('no image or label file in dir')
                logging.warning('skipped file %s: no file' % subjects[idx])
                continue
            
            list_image.append(path_image)
            list_labels.append(path_labels)
            
    return list_image, list_labels


if __name__ == '__main__':
    path_dir_1_5T = '/home/s1283/no_backup/s1283/data/1_5T'
    path_dir_3T = '/home/s1283/no_backup/s1283/data/3T'
    path_images, path_labels = get_paths(path_dir_1_5T, path_dir_3T, b_verbose = False)