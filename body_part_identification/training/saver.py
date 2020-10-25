#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from contextlib import redirect_stdout

class Saver:
    def __init__(self, config, params, save_dir, model, metrics=''):
        self.config = config
        self.params = params
        self.save_dir = save_dir
        self.save_model = model,
        self.metrics = metrics
        
    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Save config
        config_filename = self.save_dir + '/config.txt'
        with open(config_filename, 'w') as f:
            f.write(json.dumps(self.config))
        print('Configs saved.')
        
        # Save parameter
        params_filename = self.save_dir + '/params.txt'
        with open(params_filename, 'w') as f:
            f.write(json.dumps(self.params))
        print('Parameters saved.')      
        
        
        # Save metrics
        metric_filename = self.save_dir + '/metrics.txt'
        with open(metric_filename, 'w') as f:
            f.write(json.dumps(self.metrics))
        print('Metrics saved.')

        # TODO find out why save_model is tuple
        # save_model = model in train, there it is still ...
        # <class 'tensorflow.python.keras._impl.keras.engine.training.Model'>
        
        # Save model and weights
        self.save_model[0].save_weights(str(self.save_dir + '/model_weights.h5'))
        print('Keras model saved.')
        
        # Save model summary
        summary_file = self.save_dir + '/model_summary.txt'
        with open(summary_file, 'w') as f:
            with redirect_stdout(f):
                self.save_model[0].summary()
        
        # Plot all metrics
        print('Plotting metrics:\n')
        for key in self.metrics.keys():
            if 'val' in key:
                self.save_all_curves(key)
        # Close all open figures
        plt.close('all')
        print('Done plotting')

        del self.save_model
        
        
    
    def save_loss_curves(self):
        '''Save loss curves as plot (.png) in save_dir'''
        loss = np.array(self.metrics['loss'])
        val_loss = np.array(self.metrics['val_loss'])
        
        plt.figure(figsize=[8,6]) # Width, height in inches
        plt.plot(loss, 'r', linewidth=1.0)
        plt.plot(val_loss, 'b', linewidth=1.0)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
        
        x_int = []
        locs, labels = plt.xticks()
        for each in locs:
            x_int.append(int(each))
        plt.xticks(x_int)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        
        plt.savefig(str(self.save_dir + '/fig_loss.png'))
        plt.savefig(str(self.save_dir + '/fig_loss.svg'))
        print('Loss curves saved.')
        
    
    def save_accuracy_curves(self):
        '''Save accuracy curves as plot (.png) in save_dir'''
        acc = np.array(self.metrics['acc'])
        val_acc = np.array(self.metrics['val_acc'])
        
        plt.figure(figsize=[8,6]) # Width, height in inches
        plt.plot(acc, 'r', linewidth=1.0)
        plt.plot(val_acc, 'b', linewidth=1.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        
        x_int = []
        locs, labels = plt.xticks()
        for each in locs:
            x_int.append(int(each))
        plt.xticks(x_int)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        
        plt.savefig(str(self.save_dir + '/fig_accuracy.png'))
        plt.savefig(str(self.save_dir + '/fig_accuracy.svg'))
        print('Accuracy curves saved.')

        
    def save_all_curves(self, key_val):
        '''Save metricy as plot in save_dir'''
        # Strips _val
        key = key_val[4:]
        metric = np.array(self.metrics[key])
        val_metric = np.array(self.metrics[key_val])
        
        plt.figure(figsize=[8,6]) # Width, height in inches
        plt.plot(metric, 'r', linewidth=1.0)
        plt.plot(val_metric, 'b', linewidth=1.0)
    
        # For nicer looking plot
        name_dict = {'reg':'Regression', 'class': 'Classification',
                     'mean_squared_error': 'MSE', 'acc':'Accuracy',
                     'loss':'Loss', 'precision':'Precision',
                     'sensitivity':'Sensitivity', 'specificity':'Specificity'}
        
        if key[:3]=='reg':
            metric_type = name_dict['reg']
            if key[4:] in name_dict:
                name = name_dict[key[4:]]
            else:
                name = key[4:]
        elif key[:5]=='class':
            metric_type = name_dict['class']
            if key[6:] in name_dict:    
                name = name_dict[key[6:]]
            else:
                name = key[6:]
        elif key == 'loss':
            metric_type = ''
            name = name_dict[key]
        else:
            metric_type = ''
            name = key
        
        plt.legend(['Training ' + name, 'Validation ' + name], fontsize=18)
        x_int = []
        locs, labels = plt.xticks()
        for each in locs:
            x_int.append(int(each))
        plt.xticks(x_int)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel(name, fontsize=16)
     
        # Title with capitalization, filename without
        if  metric_type:
            plt.title(metric_type + ' ' + name + ' Curves', fontsize=16)
            name = name.lower()
            metric_type = metric_type.lower()
            
            plt.savefig(str(self.save_dir + '/fig_' + metric_type + '_' + name + '.png'))
            plt.savefig(str(self.save_dir + '/fig_' + metric_type + '_' + name + '.svg'))
            print(metric_type + '_' + name + ' saved')
        else:
            plt.title(name + ' Curves', fontsize=16)
            name = name.lower()
            metric_type = metric_type.lower()
            
            plt.savefig(str(self.save_dir + '/fig_' + name + '.png'))
            plt.savefig(str(self.save_dir + '/fig_' + name + '.svg'))
            print(name + ' saved')

        
if __name__ == '__main__':
    Saver(config = config, params = params, save_dir = save_dir, model = model, metrics = history.history).save()
        

    