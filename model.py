import tensorflow as tf

num_classes = 2;

# Create some wrappers for simplicity
def conv2d(x, W, b, name, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout, is_training=False):
 
    x = tf.reshape(x, shape=[-1, 110, 110, 1], name='reshape_x')
    x = tf.contrib.layers.layer_norm(x);
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], name='conv1')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.contrib.layers.layer_norm(conv1)
    conv1 = maxpool2d(conv1, k=2)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name='conv2')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.contrib.layers.layer_norm(conv2)
    conv2 = maxpool2d(conv2, k=2) 
    
  
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# In[7]:
def get_regularization_error():
    return (regularizer['wd1'] + 0.1*(regularizer['wc1'] + regularizer['wc2']))

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 1])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 1, 1])),
    'wd1': tf.Variable(tf.random_normal([28*28, 256])),
    'out': tf.Variable(tf.random_normal([256, num_classes]))
}

regularizer = {
    'wc1' : tf.nn.l2_loss(weights['wc1']),
    'wc2' : tf.nn.l2_loss(weights['wc2']),
    'wd1' : tf.nn.l2_loss(weights['wd1'])

}

biases = {
    'bc1': tf.Variable(tf.random_normal([1])),
    'bc2': tf.Variable(tf.random_normal([1])),
    'bd1': tf.Variable(tf.random_normal([1])),
    'out': tf.Variable(tf.random_normal([1]))
}

def get_weights():
    return [weights, biases, regularizer];
