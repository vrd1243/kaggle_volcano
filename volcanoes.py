import numpy as np
import pickle
import tensorflow as tf
import os, glob
import matplotlib
matplotlib.use('Agg');
from matplotlib import pyplot as plt
from data import VolcanoData 
from model import conv_net, get_regularization_error, get_weights
import gc
import imp
import pandas as pd

num_classes = 2;
beta = 1

X = tf.placeholder(tf.float32, [None, 12100], 'X_placeholder')
Y = tf.placeholder(tf.float32, [None, num_classes], 'Y_placeholder') 

learning_rate = tf.placeholder(tf.float32)

weights, biases, regularizer = get_weights(seed=10);

logits = conv_net(X, weights, biases, 1, True)
prediction = tf.nn.softmax(logits)

#entropy = tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=Y, pos_weight=10)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss_op = tf.reduce_sum(entropy)

#loss_op = tf.reduce_sum(tf.multiply((prediction - Y),(prediction - Y)), 0)
diff_sq = tf.multiply((logits - Y),(logits - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss_op + beta*get_regularization_error(regularizer));

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tp = tf.reduce_sum(tf.multiply(tf.argmax(prediction, 1), tf.argmax(Y,1)));
fp = tf.reduce_sum(tf.multiply(tf.argmax(prediction, 1), tf.argmin(Y,1)));
tn = tf.reduce_sum(tf.multiply(tf.argmin(prediction, 1), tf.argmin(Y,1)));
fn = tf.reduce_sum(tf.multiply(tf.argmin(prediction, 1), tf.argmax(Y,1)));

tss = tf.subtract(tf.divide(tp, tf.add(tp, fn)), tf.divide(fp, tf.add(fp, tn)));

def run_epoch(sess, num_steps, initial_rate, anneal=False):
    
    sess.run(tf.global_variables_initializer())
    rate = initial_rate;
    
    training_error = [];
    validation_error = [];
    training_loss = [];
    test_accuracy = [];
    test_tss = [];
    
    VObj = VolcanoData();

    for step in range(1, num_steps+1): 
        if anneal:
            rate = initial_rate * (1 - step/num_steps);
        
        batch_x, batch_y = VObj.get_next_batch(batch_size=128);
            
        l,t= sess.run([loss_op, train_op], feed_dict={X: batch_x, 
                                                      Y: batch_y, 
                                                      learning_rate: rate})
        training_loss.append(l);
        
        tps, fps, tns, fns, tss_score, acc = sess.run([tp, fp, tn, fn, tss, accuracy], feed_dict = {X: VObj.test_data,
                                                Y: VObj.test_labels_onehot});
        test_accuracy.append(acc);
        test_tss.append(tss_score);
        print(step, l, acc, tss_score);
        #print(tps / (tps + fns), fns / (tps + fns), fps / (fps + tns), tns / (fps + tns));
              
    training_loss = np.array(training_loss); 
    
    plt.figure();
    plt.plot(training_loss);
    plt.savefig('training.png');

    plt.figure();
    plt.plot(test_accuracy);
    plt.savefig('accuracy.png');

    plt.figure();
    plt.plot(test_tss);
    plt.savefig('tss.png');

with tf.Session() as sess:
    run_epoch(sess, 1000, 0.01, True);
