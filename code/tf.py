__author__ = 'naveen'
# Libs import
import numpy as np
import tensorflow as tf
import random

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def tensorFlowNN(X_tr, y_tr, X_te, y_te):
    D = X_tr.shape[1]
    num_classes = np.unique(y_tr).shape[0];
    N_tr = X_tr.shape[0]
    N_te = X_te.shape[0]

    # Label to map
    y_map = np.zeros((num_classes,num_classes))
    for i in range (num_classes):
        y_map[i,i] = 1

    y_tr_m = y_map[y_tr]
    y_te_m = y_map[y_te]

    """
    # Single Layer Neural Network
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, D])
    y_ = tf.placeholder("float", shape=[None, num_classes])

    W = tf.Variable(tf.zeros([D,num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    sess.run(tf.initialize_all_variables())

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    all_idx = np.arange(0,N_tr)

    for i in range(1000):
        print i
        idx = random.sample(all_idx,100)
        X_tmp = X_tr[idx,:]
        y_tmp = y_tr_m[idx,:]
        sess.run(train_step, feed_dict={x: X_tmp, y_: y_tmp})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Training Accuracy = %f" % sess.run(accuracy, feed_dict={x: X_tr, y_: y_tr_m})

    print "Test Accuracy = %f" % sess.run(accuracy, feed_dict={x: X_te, y_: y_te_m})

    """

    # CNN

    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, D])
    y_ = tf.placeholder("float", shape=[None, num_classes])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,20,20,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    all_idx = np.arange(0,N_tr)

    for i in range(1000):
      idx = random.sample(all_idx,100)
      X_tmp = X_tr[idx,:]
      y_tmp = y_tr_m[idx,:]
      print X_tmp[0,:]
      print y_tmp[0,:]
      print 'here'

      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: X_tmp, y_: y_tmp, keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
      train_step.run(feed_dict={x: X_tr[idx,:], y_: y_tr_m[idx,:], keep_prob: 0.5})
      if i==1:
        exit()

    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: X_te, y_: y_te_m, keep_prob: 1.0})

    exit()