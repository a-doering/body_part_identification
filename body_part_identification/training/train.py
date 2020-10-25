import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe)))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import time
import numpy as np
from collections import defaultdict

from data_generator import DataGenerator
from model_hybrid import build_model
from prepare_data import prepare_data
from saver import Saver
import custom_metrics as cm

import tensorflow as tf
import gc
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config = config))

def start_training(config, params, save_dir):
    '''
    Starts a training
    '''
    print('\n')
    print('#'*100)
    print('Next training starts:')
    print('#'*100)
    print(config, '\n')
    print(params, '\n')
    
    
    time_train = time.time()
    path_dir_1_5T = '/home/s1283/no_backup/s1283/data/1_5T'
    path_dir_3T = '/home/s1283/no_backup/s1283/data/3T'
    n_patients, dict_images, dict_labels, dict_patch_pos, keys = prepare_data(params['patch_size'],params['overlap'], path_dir_1_5T, path_dir_3T)
    
    
    # For consistent validation data and reproducability
    np.random.seed(config['seed'])
    np.random.shuffle(keys)
    
    # Split data in training and validation data
    split = int(n_patients*config['train_val_split'])
    keys_val = keys[:split]
    keys_train = keys[split:]
    
     # Create dicts to reduce size of dicts that are passed to data generator
    dict_images_train = {key:dict_images[key] for key in keys_train}
    dict_labels_train = {key:dict_labels[key] for key in keys_train}
    dict_patch_pos_train = {key:dict_patch_pos[key] for key in keys_train}
     
    dict_images_val = {key:dict_images[key] for key in keys_val}
    dict_labels_val = {key:dict_labels[key] for key in keys_val}
    dict_patch_pos_val = {key:dict_patch_pos[key] for key in keys_val}
     
    gen_train = DataGenerator(dict_images_train, dict_labels_train, dict_patch_pos_train, keys_train, **params)
    gen_val = DataGenerator(dict_images_val, dict_labels_val, dict_patch_pos_val, keys_val, **params)    

    # Build model, train
    model = build_model(params['patch_size'], params['n_classes'])
    print(model.summary())
    
    # Python functions are not JSON serializable, therefore the strings
    # From the config have to be changed to functions
    custom_metrics = {'precision':cm.precision, 'sensitivity':cm.sensitivity, 'specificity':cm.specificity}
    used_metrics = defaultdict(list)
    for metric_type, metric_functions in config['metrics'].items():
        for function in metric_functions:
            if function in custom_metrics.keys():
                used_metrics[metric_type].append(custom_metrics[function])
            else:
                used_metrics[metric_type].append(function)
        
    model.compile(loss=config['loss'], 
                  optimizer=config['optimizer'], 
                  metrics=used_metrics, 
                  loss_weights=config['loss_weights'])

    history = model.fit_generator(generator = gen_train,
                        use_multiprocessing=True,
                        workers=5,
                        epochs =config['epochs'],
                        validation_data = gen_val)

    # Save everything
    print(save_dir)
    Saver(config = config, 
          params = params, 
          save_dir = save_dir, 
          model = model, 
          metrics = history.history).save()
    
    print("\n\nTraining time taken: ", time.time()-time_train)

# =============================================================================
# This can be used as a primitive hyperparameter tuner
# =============================================================================
if __name__ == '__main__':
    time0 = time.time()
    
    # Parameters for data preparation
    params = {'batch_size':1000,
              'patch_size':[1,60,16],
              'n_channels':1,
              'n_classes':5,
              'n_patches':500,
              'overlap':0.6,
              'shuffle':True}
    
    # Training configuration
    config = {'epochs':500,
              'loss':{'class':'categorical_crossentropy', 'reg':'MSE'},
              'optimizer':'Adam',
              'learn_rate':0.001,
              'metrics':{'class':['accuracy', 'precision', 'sensitivity', 'specificity'], 'reg':['MSE']},
              'train_val_split':0.1, # e.g. 0.2 -> 80% train, 20% validation
              'loss_weights':[4,1],
              'seed':27}
    
    save_path = '/home/s1283/no_backup/s1283/hyperparameter_tuning_01/training_'
    folder_num = 0
    
    # Fot grid search, create lists of values that can be executed one 
    # after another in loops
    n_patches = [250]
    patch_sizes = [[1,40,16]]
    loss_weights = [[4,1]]
    overlap = [0.9]

    tuning = False
    if tuning:

        for num_p in n_patches:
            params['n_patches'] = num_p    
            for ps in patch_sizes:
                params['patch_size'] = ps
                for lw in loss_weights:
                    config['loss_weights'] = lw
                    for ol in overlap:
                        params['overlap'] = config['overlap'] = ol

    
                        # Make sure that the directory does not exist prior
                        save_dir = save_path + f'{folder_num:02}'
                        while os.path.isdir(save_dir):
                            folder_num +=1
                            save_dir = save_path + f'{folder_num:02}'
                        
                        print(save_dir)
                        start_training(config, params, save_dir)
                        gc.collect()
                        folder_num += 1
    else:
        # Make sure that the directory does not exist prior
        save_dir = save_path + f'{folder_num:02}'
        while os.path.isdir(save_dir):
            folder_num +=1
            save_dir = save_path + f'{folder_num:02}'
        
        print(save_dir)
        start_training(config, params, save_dir)
            
    print("\n\nTotal time taken: ", time.time()-time0)