""" Build a model to classify the images in the sprace dataset
"""
from absl import flags
from sprace_dataset import load_dataset

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
                    default="/tmp/sprace_model",
                    help="Directory where model is stored.")
FLAGS = flags.FLAGS


def cnn_model_fn(features, labels, mode):
    """ Model function for CNN
    """
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(
        inputs=dropout,
        units=3)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predictions['classes'])}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def main(_):
    sprace = load_dataset()
    train_data = sprace.train.images
    train_labels = sprace.train.labels
    eval_data = sprace.validation.images
    eval_labels = sprace.validation.labels

    sprace_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=FLAGS.model_dir)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)
    sprace_classifier.train(
        input_fn=train_input_fn,
        steps=FLAGS.steps)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = sprace_classifier.evaluate(
        input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
