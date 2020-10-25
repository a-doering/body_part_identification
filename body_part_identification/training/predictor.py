import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from training import model_hybrid
from medio import read_image
from util import patches as pat

def plot_probability_maps(probability_maps, save_path,n_classes=5):
    '''Plot all n_classes probability maps'''
    
    landmark_dict = {0:'Wrists', 1:'Shoulders', 2:'Liver_dome', 3:'Hips', 4:'Heels'}
    
    for i in range(n_classes):
        plt.close('all')
        fig, ax = plt.subplots()
        probability_map = ax.pcolor(np.rot90(prob_maps[i], k=1, axes=(1,0)))
        cbar = plt.colorbar(probability_map)
        cbar.ax.set_ylim(0,1)
        cbar.ax.set_ylabel('Probability')
        plt.title('Probability Map of Class: ' + landmark_dict[i])        
        plt.ylabel('HF Slice Number')
        plt.xlabel('LR Slice Number')
        
        save_name = save_path + '/probability_map_' + landmark_dict[i].lower() + '.png'
        plt.savefig(save_name, bbox_inches ='tight')#, pad_inches=0)
        
def get_thresholds(decision_map, n_classes=5):
    '''Calculates thresholds of predicted decision map'''
    row_max = np.zeros(decision_map.shape[1], dtype='int32')
    class_cnt = np.zeros(n_classes,dtype='int32')
    
    for i in range(decision_map.shape[1]):
        class_cnt = np.zeros(n_classes,dtype='int32')
        for j in range(decision_map.shape[0]):
            class_cnt[decision_map[j][i]]+=1
        row_max[i] = np.argmax(class_cnt)
    
    thresholds = np.zeros(n_classes-1,dtype='int32')
    idx = 0
    for i in range(decision_map.shape[1]-2):
        if row_max[i] != row_max[i+1] and row_max[i+1]==row_max[i+2]:
            thresholds[idx] = i+1
            idx+=1
            if idx==n_classes-1:
                return thresholds

    return thresholds

def combine(image, heatmap, alpha=0.4, display=False, save_path=None, cmap='viridis', axis='on', verbose=False, rotate=False, pred_thresholds=[], real_thresholds=[]):
    '''Combine image with heatmap, plot and save'''
    aspect = 0.1
    if real_thresholds==[]:
        real_thresholds = np.zeros(pred_thresholds.shape)
    
    if rotate:
        image = np.rot90(image, k=1, axes=(1,0))
        heatmap = np.rot90(heatmap, k=1, axes=(1,0))
        aspect = 1.0/aspect
        
    # Discrete color scheme
    cMap = ListedColormap(['midnightblue', 'darkslateblue', 'darkcyan', 'olive', 'goldenrod'])

    # Display
    fig, ax = plt.subplots()
    image = ax.pcolormesh(image)
    heatmap = ax.pcolor(heatmap, alpha=alpha, cmap=cMap)
    ax.set_aspect(aspect=aspect)

    cbar = plt.colorbar(heatmap)    
    landmark_dict = {0:'Wrists', 1:'Shoulders', 2:'Liver_dome', 3:'Hips', 4:'Heels'}
    cbar.ax.set_ylabel('Class')
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(landmark_dict):
        cbar.ax.text(.5, (2 * j + 1) / 10.0, landmark_dict[j], ha='center', va='center', rotation=90)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.invert_yaxis()

    real_thresholds = np.flip(real_thresholds,0)
    for t in range(pred_thresholds.size):
        plt.axhline(y=pred_thresholds[t], color='r', ls='dashed', label='Predicted Threshold')
        plt.axhline(y=int(real_thresholds[t]), color='w', ls='dotted', label='Real Threshold')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles = handles[:2], labels=['Predicted Threshold','Real Threshold'],loc='upper center', bbox_to_anchor= (0.5,-0.1), ncol=2)
#    ax.legend.get_frame().set_facecolor('0.5')
    
    plt.title('Decision Map')
    plt.ylabel('HF Slice Number')
    plt.xlabel('LR Slice Number')
    
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    
    if display:
        plt.show()
    
    if save_path is not None:
        if verbose:
            print('Heatmap with image saved at ', save_path)
        save_name = save_path + '/heatmap_' + str(alpha)
        plt.savefig(save_name  + '.png', bbox_inches ='tight', pad_inches=0)
        plt.savefig(save_name  + '.svg', bbox_inches ='tight', pad_inches=0)

if __name__ == '__main__':
    # Loading the model with weights
    src_dir = "/home/s1283/no_backup/s1283/hyperparameter_tuning_01/training_05"#prev results/training37
 
    # Load config and params
    with open(src_dir + '/config.txt', 'r') as f:
        config = json.load(f)
    with open(src_dir + '/params.txt', 'r') as f:
        params = json.load(f)
        
    n_classes = params['n_classes']
    patch_size = params['patch_size']
     
    model = model_hybrid.build_model(patch_size, params['n_classes'])
    model.load_weights(src_dir + '/model_weights.h5')
    model.compile(loss=config['loss'], 
                    optimizer=config['optimizer'], 
                    #metrics=config['metrics'], 
                    loss_weights=[4,1])

    image_path = '/home/s1283/no_backup/s1283/data/test_data_mixed/TULIP_1645_GZ/rework.mat'
    label_path = '/home/s1283/no_backup/s1283/data/test_data_mixed/TULIP_1645_GZ/labels_TULIP_1645_GZ_.mat'

    image = read_image.read_mat_image(image_path)
    labels = read_image.read_mat_labels(label_path)
    coronal_slice_idx = image.shape[0]//2
    
    # Number of slices per side around the middle coronal slice
    slices = 3
    image = image[coronal_slice_idx-slices:coronal_slice_idx+slices]
    patch_pos = pat.compute_patch_indices(image.shape, patch_size ,0.9)
    
    patches = np.zeros((len(patch_pos), 1, patch_size[1], patch_size[2]),dtype='float32')
    for i,pos in enumerate(patch_pos,0):
        patches[i] = image[pos[0]:(pos[0]+patch_size[0]),pos[1]:(pos[1]+patch_size[1]), pos[2]:(pos[2]+patch_size[2])]
    
    prediction_prob, prediction_mse = model.predict(patches)
    del patches
    
    prob_maps = np.zeros([n_classes, image.shape[1], image.shape[2]])
    decision_map = np.zeros([n_classes, image.shape[1], image.shape[2]])
    voxel_appeared = np.zeros([image.shape[1], image.shape[2]], dtype=int)

    for i,pos in enumerate(patch_pos,0):
        decision_map[np.argmax(prediction_prob[i]),pos[1]:(pos[1]+patch_size[1]), pos[2]:(pos[2]+patch_size[2])]+=1
        voxel_appeared[pos[1]:(pos[1]+patch_size[1]), pos[2]:(pos[2]+patch_size[2])] += 1
        
        for c in range(n_classes): 
            prob_maps[c,pos[1]:(pos[1]+patch_size[1]), pos[2]:(pos[2]+patch_size[2])] += prediction_prob[i][c]
            
    # Division needed as some voxel appear more often   
    prob_maps = np.divide(prob_maps, voxel_appeared, out=np.zeros_like(prob_maps), where=voxel_appeared!=0)
    absolute_decision_map = np.argmax(decision_map, axis=0)
#%%
    save_path='/home/s1283/no_backup/s1283/heatmap_plots_test'
    plot_probability_maps(prob_maps, save_path)
    
    pred_thresholds = get_thresholds(absolute_decision_map)
    real_thresholds = np.zeros(n_classes-1)
    
    for i in range(n_classes-1):
        real_thresholds[i] = (labels[i]+labels[i+1])//2
    combine(image[image.shape[0]//2], absolute_decision_map, rotate=True, 
            save_path=save_path, pred_thresholds=pred_thresholds,real_thresholds=real_thresholds)
