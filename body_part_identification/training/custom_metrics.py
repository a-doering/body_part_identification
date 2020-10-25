import keras.backend as K

def precision(y_true, y_pred):
    '''Metric: true positives / (true positives + false positives)'''
    
    neg_y_true = 1 - y_true
    tp = K.sum(y_true[...]*y_pred[...])
    fp = K.sum(neg_y_true[...] * y_pred[...])
    precision = tp / (tp + fp + K.epsilon())
    return precision
    
def sensitivity(y_true, y_pred): # also called recall
    '''Metric: true positives / (true positives + false negatives)'''
    
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true[...]*y_pred[...])
    fn = K.sum(y_true[...]*neg_y_pred[...])
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity
    
def specificity(y_true, y_pred):
    '''Metric: true negatives / (true negatives + false positives)'''

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true[...] * y_pred[...])
    tn = K.sum(neg_y_true[...] * neg_y_pred[...])
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

if __name__ == '__main__':
    import numpy as np
    # Tests
    y_true = np.array([[1,0,0,0,0],[1,0,1,0,0], [0,0,0,0,1]])
    y_pred = np.array([[0,0,0,0,0],[1,0,0,1,0], [1,1,1,0,1]])
    
    pre = K.eval(precision(K.constant(y_true), K.constant(y_pred)))
    sen = K.eval(sensitivity(K.constant(y_true), K.constant(y_pred)))
    spe = K.eval(specificity(K.constant(y_true), K.constant(y_pred)))
    
    y_true = np.array([[1,0,0,0,0],[1,0,1,0,0], [0,0,0,0,1]])
    y_pred = np.array([[1,0,0,0,0],[1,0,0,1,0], [1,1,1,0,1]])