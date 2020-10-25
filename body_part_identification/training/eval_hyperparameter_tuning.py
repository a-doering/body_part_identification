from glob import glob
import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt

def print_results_table(path_trainings):
    '''
    Print a latex table of the results to a .txt file
    '''
    n_val = 7
    results = np.zeros([len(path_trainings), 2*n_val])
    
    for idx, path in enumerate(path_trainings,0):
        
        with open(path + '/metrics.txt', 'r') as f:
            metrics = json.load(f)
        keys = ['val_class_loss', 'val_class_acc', 'val_class_precision', 'val_class_sensitivity','val_class_specificity', 'val_reg_loss', 'val_reg_mean_squared_error']
        for idx_key, key in enumerate(keys,0):
            results[idx][idx_key] = np.mean(metrics[key][-100:])
            results[idx][idx_key+n_val] = np.std(metrics[key][-100:])
    
    header = ['Training','$Loss_{Class}$', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity',  'MSE']#'$Loss_{Reg}$',
    
    orig_stdout = sys.stdout
    with open('results_table.txt', 'w') as f:
        sys.stdout = f
        for h in header:
            print(h, end='')
            if h != header[-1]:
                print(' & ', end='')
            else:
                print(' \\\\ \n\\hline')
        for training_idx, line in enumerate(results,0):
            print('H'+f'{training_idx:02}'+ ' & ', end='')
            for idx in range(n_val-1): #-1 as MSE == Loss_Reg
                print(f'{line[idx]:.3f}' + ' $\pm$ ' + f'{line[idx+n_val]:.3f}', end='')
                if idx != n_val-1:
                    print(' & ', end='')
            if training_idx != len(results)-1:
                print(' \\\\ \n', end='')
    
        print('')
        print('%argmax: ' + str(np.argmax(results, axis=0)))
        print('%argmin: ' + str(np.argmin(results, axis=0)))
    sys.stdout = orig_stdout        
    return results[:,:n_val-1]

def print_config_table(path_trainings):
    '''
    Print a latex table of configurations to a .txt file
    '''
    header = ['Training', '$N_{Patches pp.}$', 'Patch Size', 'Overlap', 'Loss Weights(C:R)']
    
    # Create list of lists
    configs_used = []
    for i in range(len(path_trainings)):
        configs_used.append([])
        for j in range(len(header)):
            configs_used[i].append([])
    
    # Load data
    for idx, path in enumerate(path_trainings,0):
        with open(path + '/params.txt', 'r') as f:
            params = json.load(f)
        with open(path + '/config.txt', 'r') as f:
            config = json.load(f)

        configs_used[idx][0] = params['n_patches']
        configs_used[idx][1] = params['patch_size']
        configs_used[idx][2] = params['overlap']
        configs_used[idx][3] = config['loss_weights']
    
    # Print data 
    orig_stdout = sys.stdout
    with open('hyperparameter_table.txt', 'w') as f:
        sys.stdout = f
        for h in header:
            print(h, end='')
            if h != header[-1]:
                print(' & ', end='')
            else:
                print(' \\\\ \n\\hline')
#        for idx in range(len(configs_used)):
        for training_idx, line in enumerate(configs_used,0):
            print('H'+f'{training_idx:02}'+ ' & ', end='')
            for idx in range(len(header)):
                print(line[idx], end='')
                if idx != len(header) -1:
                    print(' & ', end='')
            if training_idx != len(configs_used)-1:
                print(' \\\\ \n', end='')             
    sys.stdout = orig_stdout            
    

if __name__ == '__main__':
    src_dir = '/home/s1283/no_backup/s1283/hyperparameter_tuning_01/'
    path_trainings = glob(os.path.join(src_dir, '*/'))
    path_trainings.sort()
    results = print_results_table(path_trainings)
    configs = print_config_table(path_trainings)

#%%
    # Create boxplots for hyperparameter training
    loss_class = results[:,0]
    mse = results[:,-1]
    metrics_class = results[:,1:-1]

    fig, ax1 = plt.subplots(figsize=(5,5))
    bp1 = ax1.boxplot(100*metrics_class,showmeans=True)
    ax1.set_ylim([0,100])
    ax1.set_ylabel('Percentage')
    
    ax1.set_xticklabels(['Accuracy', 'Precision', 'Sensitivity', 'Specificity'])
    ax1.set_title('Classification Metrics')
    plt.savefig('classification_boxplots.png', bbox_inches ='tight')

    fig, ax2 = plt.subplots(figsize=(2,5))
    bp2 = ax2.boxplot(mse, showmeans=True)
    ax2.set_xticklabels(['MSE'])
    ax2.set_ylabel('Difference in Slices')
    ax2.set_title('Regression Metric')
    plt.tight_layout()

    plt.savefig('regression_boxplots.png',bbox_inches ='tight')

    