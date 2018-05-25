""" Build a model to classify the images in the custom dataset
"""
from absl import flags
from custom_dataset import load_dataset

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_integer("steps",
                     default=100,
                     help="Number of steps.")
flags.DEFINE_string("model_dir",
                    default="/tmp/model_dir",
                    help="Directory where model is stored.")
FLAGS = flags.FLAGS


def dnn(inputs, mode):
    """ Design deep neural network
    """
    conv1 = tf.layers.conv2d(
        inputs, 32, 5,
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1, 2, 2)
    conv2 = tf.layers.conv2d(
        pool1, 64, 5,
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2, 2, 2)
    dense = tf.layers.dense(
        tf.reshape(pool2, [-1, 7*7*64]),
        1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        dense, rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(
        dropout, 3)
    return logits


def model_fn(features, labels, mode):
    """ Model function
    """
    inputs = tf.reshape(features['x'], [-1, 28, 28, 1])
    logits = dnn(inputs, mode)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate).minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={
            'accuracy': tf.metrics.accuracy(labels, predictions['classes'])})


def main(_):
    """ Load dataset.
        Execute custom estimator.
    """
    dataset = load_dataset()

    classifier = tf.estimator.Estimator(model_fn, FLAGS.model_dir)

    classifier.train(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'x': dataset.train.images},
            y=dataset.train.labels,
            batch_size=FLAGS.batch_size,
            num_epochs=None,
            shuffle=True),
        steps=FLAGS.steps)

    eval_results = classifier.evaluate(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'x': dataset.validation.images},
            y=dataset.validation.labels,
            num_epochs=1,
            shuffle=False))

    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
