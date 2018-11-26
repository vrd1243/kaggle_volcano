import numpy as np
import pickle
import tensorflow as tf
import os, glob
import matplotlib
matplotlib.use('Agg');
from matplotlib import pyplot as plt
import gc
import imp
import pandas as pd

class VolcanoData: 

  def __init__(self):
    self.train_data = pd.read_csv('../../data/volcanoes/train_images.csv', header=None);
    train_labels = pd.read_csv('../../data/volcanoes/train_labels.csv');
    self.train_labels_onehot = np.zeros((train_labels.shape[0], 2));
    self.train_labels_onehot[:,0] = (train_labels['Volcano?'].values == 0)*1;
    self.train_labels_onehot[:,1] = (train_labels['Volcano?'].values == 1)*1;

    self.pos_idx = np.where(train_labels['Volcano?'].values == 1)[0];
    self.neg_idx = np.where(train_labels['Volcano?'].values == 0)[0];
    
    self.ratio = self.pos_idx.shape[0] / (self.pos_idx.shape[0] + self.neg_idx.shape[0]);
    #self.ratio = 0.5;

    self.test_data = pd.read_csv('../../data/volcanoes/test_images.csv', header=None);
    test_labels = pd.read_csv('../../data/volcanoes/test_labels.csv');
    self.test_labels_onehot = np.zeros((test_labels.shape[0], 2));
    self.test_labels_onehot[:,0] = (test_labels['Volcano?'].values == 0)*1;
    self.test_labels_onehot[:,1] = (test_labels['Volcano?'].values == 1)*1;
        
  def get_next_batch(self, batch_size):
      
      np.random.shuffle(self.pos_idx);
      np.random.shuffle(self.neg_idx);
      num_pos = int(self.ratio*batch_size);  
      num_neg = batch_size - num_pos; 

      idx = np.concatenate((self.pos_idx[:num_pos], self.neg_idx[:num_neg]),
                            axis=0); 
      return self.train_data.iloc[idx], self.train_labels_onehot[idx]; 
  

VData = VolcanoData();

data, labels = VData.get_next_batch(128); 
print(data.shape);
print(labels.shape);
