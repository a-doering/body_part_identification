from scipy.io import loadmat
import numpy as np

def read_mat_image(path):
    '''Convert mat tensor to numpy.ndarray'''
    tensor = loadmat(path, variable_names = ['img'])
    tensor = np.array(tensor['img'], np.float32)
    return tensor
    
    
def read_mat_labels(path):
    '''Convert matlab labels to numpy.ndarray'''
    labels = loadmat(path)
    labels = labels['auxsave2']
    
    label_names = ['wrist', 'shoulder', 'heartEnd', 'hip', 'heel']
    label = np.zeros(5)
    for i in range(5):
        label[i] = labels[0][label_names[i]][0][0][0]
        
    label = label.astype(np.int32)
    return label