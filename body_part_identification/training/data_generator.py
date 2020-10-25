import numpy as np
from util import patches
from util import convert_label
from tensorflow.python.keras.utils import Sequence

class DataGenerator(Sequence):
    '''Generate data for keras'''
    def __init__(self, images, labels, patch_pos, keys, batch_size = 500,
                 patch_size = [40,16], n_channels = 1, n_classes = 5,
                 n_patches = 100, overlap =0.6, shuffle = True):
        '''Initialization'''
        self.images = images
        self.labels = labels
        self.patch_pos = patch_pos
        self.keys = keys
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_patches = n_patches
        self.overlap = overlap
        self.shuffle = shuffle
        self.on_epoch_end()
            
    def __getitem__(self,index):
        '''Generate one batch of data'''
        # Generate indices of the batch
        list_keys_temp = self.keys[index*self.batch_size//self.n_patches:(index+1)*self.batch_size//self.n_patches]

        # Generate data
        X, y, y_reg = self.__data_generation(list_keys_temp)
        return X, {'class':y, 'reg':y_reg}
    
    def on_epoch_end(self):
        '''Update indices after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.keys)
    
    def __data_generation(self, list_keys_temp):
        '''Generate data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, *self.patch_size))
        y = np.empty((self.batch_size, self.n_classes), dtype='int32')
        y_reg = np.empty((self.batch_size, 1))
        
        for idx, key in enumerate(list_keys_temp,0):
            # Draw n_patches from the all possible patch positions in patch_pos
            patch_pos_temp = patches.draw_num_patches(self.patch_pos[key], self.n_patches)
            # Extract patches
            X[idx*self.n_patches:(idx+1)*self.n_patches] = patches.extract_patches(self.images[key],patch_pos_temp, self.n_patches, self.n_channels, self.patch_size)
            # Create one hot encoded vector
            y[idx*self.n_patches:(idx+1)*self.n_patches] = convert_label.make_class_label(self.labels[key], patch_pos_temp, self.patch_size)
            # Create regression label
            y_reg[idx*self.n_patches:(idx+1)*self.n_patches] = convert_label.make_reg_label(self.labels[key], patch_pos_temp, self.patch_size)
            
        return X, y, y_reg
    
    def __len__(self):
        '''Denotes number of batches per epoch'''
        return int(np.floor(len(self.keys)*self.n_patches/self.batch_size))


        
    
    
                
    