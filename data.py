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

def read_data():
  train_data = pd.read_csv('../volcanoes/train_images.csv', header=None);
  train_labels = pd.read_csv('../volcanoes/train_labels.csv');
  train_labels_onehot = np.zeros((train_labels.shape[0], 2));
  train_labels_onehot[:,0] = (train_labels['Volcano?'].values == 0)*1;
  train_labels_onehot[:,1] = (train_labels['Volcano?'].values == 1)*1;

  test_data = pd.read_csv('../volcanoes/test_images.csv', header=None);
  test_labels = pd.read_csv('../volcanoes/test_labels.csv');
  test_labels_onehot = np.zeros((test_labels.shape[0], 2));
  test_labels_onehot[:,0] = (test_labels['Volcano?'].values == 0)*1;
  test_labels_onehot[:,1] = (test_labels['Volcano?'].values == 1)*1;
  
  return [train_data.values, train_labels_onehot, test_data.values, test_labels_onehot];

def get_next_batch(data, labels, batch_size):
    idx = np.arange(data.shape[0]);
    np.random.shuffle(idx);
    idx = np.array(idx[:batch_size]);
    
    return data[idx], labels[idx]; 

[train_data, train_labels, test_data, test_labels] = read_data();

data, labels = get_next_batch(train_data, train_labels, 128); 
print(train_labels.shape, test_labels.shape);
