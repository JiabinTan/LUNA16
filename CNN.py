# -*-coding:utf-8-*-
import tensorflow as tf
import tf.keras.backend as K
from tensorflow.contrib.layers.python.layers import batch_norm

class CNN_3d:
    def __init__(self, xs, is_training, keep_prob):
        self.xs = xs
        self.is_training = is_training
        self.keep_prob = keep_prob

    # renew Weight
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # renew bias
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # converlution
    def conv3d(self, x, W):
        return tf.nn.conv3d(x, W, [1,1,1,1,1], padding='SAME')

    # pooling
    def max_pool_2x2x2(self, x):
        return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

    # BN
    def batch_normal_layer(self, value, is_training=False):
        if is_training is True:
            # training
            return batch_norm(inputs=value, decay=0.999, is_training=True)
        else:
            # test
            return batch_norm(inputs=value, decay=0.999, is_training=False)

    def cnn(self):
        ## conv1_1 inception 1 layer ##
        W_conv1_1 = self.weight_variable([3,3,3,1,32]) # patch 3x3x3, in size 1, out size 32
        b_conv1_1 = self.bias_variable([32])
        h_conv1_1 = tf.nn.relu(self.batch_normal_layer(self.conv3d(self.xs, W_conv1_1) + b_conv1_1, self.is_training))
        h_pool1_1 = self.max_pool_2x2x2(h_conv1_1)

        ## conv2_1 inception 1 layer ##
        W_conv2_1 = self.weight_variable([3,3,3,32,32])
        b_conv2_1 = self.bias_variable([32])
        h_conv2_1 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool1_1, W_conv2_1) + b_conv2_1, self.is_training))
        h_pool2_1 = self.max_pool_2x2x2(h_conv2_1)
        #if(self.is_training):
        #    h_pool2_1_drop = tf.nn.dropout(h_pool2_1, self.keep_prob)

        ## conv1_2 inception 2 layer ##
        W_conv1_2 = self.weight_variable([5,5,5,1,32])
        b_conv1_2 = self.bias_variable([32])
        h_conv1_2 = tf.nn.relu(self.batch_normal_layer(self.conv3d(self.xs, W_conv1_2) + b_conv1_2, self.is_training))
        h_pool1_2 = self.max_pool_2x2x2(h_conv1_2)

        ## conv2_2 inception 2 layer ##
        W_conv2_2 = self.weight_variable([5,5,5,32,32])
        b_conv2_2 = self.bias_variable([32])
        h_conv2_2 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool1_2, W_conv2_2) + b_conv2_2, self.is_training))
        h_pool2_2 = self.max_pool_2x2x2(h_conv2_2)
        #if(self.is_training):
        #    h_pool2_2_drop = tf.nn.dropout(h_pool2_2, self.keep_prob)

        ## concatenate ##
        h_pool2 = K.concatenate([h_pool2_1, h_pool2_2], axis=0)
        if(self.is_training):
            h_pool2= tf.nn.dropout(h_pool2, self.keep_prob)
        ## conv3 layer ##
        W_conv3 = self.weight_variable([3,3,3,64,64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool2, W_conv3) + b_conv3, self.is_training))
        h_pool3 = self.max_pool_2x2x2(h_conv3)

        ## conv4 layer ##
        W_conv4 = self.weight_variable([3,3,3,64,64])
        b_conv4 = self.bias_variable([64])
        h_conv4 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool3,W_conv4) + b_conv4, self.is_training))
        h_pool4 = self.max_pool_2x2x2(h_conv4)
        if(self.is_training):
            h_pool4_drop = tf.nn.dropout(h_pool4, self.keep_prob)

        ## conv5 layer ##
        W_conv5 = self.weight_variable([3,3,3,64,128])
        b_conv5 = self.bias_variable([128])
        h_conv5 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool4_drop, W_conv5)+b_conv5, self.is_training))
        h_pool5 = self.max_pool_2x2x2(h_conv5)

        ## conv6 layer ##
        W_conv6 = self.weight_variable([3,3,3,128,128])
        b_conv6 = self.bias_variable([128])
        h_conv6 = tf.nn.relu(self.batch_normal_layer(self.conv3d(h_pool5, W_conv6)+b_conv6, self.is_training))
        h_pool6 = self.max_pool_2x2x2(h_conv6)
        if(self.is_training):
            h_pool6_drop = tf.nn.dropout(h_pool6, self.keep_prob)

        ## func1 layer ##
        W_fc1 = self.weight_variable([128, 512])
        b_fc1 = self.bias_variable([512])
        h_pool6_drop_flat = tf.reshape(h_pool6_drop, [-1, 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool6_drop_flat, W_fc1)+b_fc1)
        if(self.is_training):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        ## func2 layer ##
        W_fc2 = self.weight_variable([512, 512])
        b_fc2 = self.bias_variable([512])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
        if(self.is_training):
            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        ## func3 layer ##
        W_fc3 = self.weight_variable([512, 2])
        b_fc3 = self.bias_variable([2])
        prediction = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3)+b_fc3)
        model = tf.keras.Model(inputs=self.xs, outputs=prediction)
        return model