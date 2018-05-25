""" Create models using TensorFlow
"""
import tensorflow as tf


def dnn(inputs, training):
    """ Deep convolutional neural network
    """
    conv1 = tf.layers.conv2d(
        inputs,
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2)
    conv2 = tf.layers.conv2d(
        pool1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=2,
        strides=2)
    dense = tf.layers.dense(
        tf.reshape(pool2, [-1, 7*7*64]),
        units=1024,
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        dense, rate=0.4,
        training=training)
    logits = tf.layers.dense(
        dropout,
        units=3,
        activation=None)
    return logits
