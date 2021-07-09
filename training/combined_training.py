import sys
sys.path.append('../')

from models import combined_model
from data import sample_generators
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

def combine_training(x_truth, y_truth, dropout, learning_rate, epochs, display_step=2000):
    """
    Generic training of a Combined (uncertainty) network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder
    """   
    model = combined_model.CombinedModel(dropout)
    data_shape = list(np.shape(x_truth))
    data_shape[0]=None
    model.build(tuple(data_shape))
    
    def loss_fcn(ground_truth, prediction, log_variance):
        return tf.reduce_sum(
            0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(ground_truth - prediction))
            + 0.5 * log_variance
        )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    @tf.function
    def train_step(inputs, ground_truth):
        with tf.GradientTape() as tape:
            prediction, log_variance = model(inputs)
            loss = loss_fcn(ground_truth, prediction, log_variance)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, prediction, log_variance
    
    for epoch in range(epochs):
        loss, prediction, log_variance = train_step(x_truth, y_truth)
        
        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            print("Loss {}".format(loss))
            print("================")
            
    print("Training done!")
    
    return model


def combined_training_old(x_truth, y_truth, dropout, learning_rate, epochs, display_step=2000):
    """
    Generic training of a Combined (uncertainty) network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder
    """
    tf.reset_default_graph()
    x_placeholder = tf.placeholder(tf.float32, [None, 1])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])
    dropout_placeholder = tf.placeholder(tf.float32)

    prediction, log_variance = combined_model.combined_model(x_placeholder, dropout_placeholder)

    tf.add_to_collection("prediction", prediction)
    tf.add_to_collection("log_variance", log_variance)

    loss = tf.reduce_sum(0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(y_placeholder - prediction))
                         + 0.5 * log_variance)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        feed_dict = {x_placeholder: x_truth.reshape([-1, 1]),
                     y_placeholder: y_truth.reshape([-1, 1]),
                     dropout_placeholder: dropout}

        sess.run(train, feed_dict=feed_dict)

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            current_loss = sess.run(loss, feed_dict=feed_dict)
            print("Loss {}".format(current_loss))
            print("================")

    print("Training done")

    return sess, x_placeholder, dropout_placeholder
