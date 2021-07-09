import sys
sys.path.append('../')

from scipy.stats import bernoulli
from models import bootstrap_model
from data import sample_generators

import tensorflow as tf
import numpy as np


import matplotlib.pyplot as plt

def bootstrap_training(x_truth, y_truth, dropout, learning_rate, epochs, n_heads, display_step=2000):
    """
    Generic training of Boostrap Network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param n_heads: Number of heads for trained Network
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder, mask_placeholder
    """
    # This placeholder holds the mask indicating which heads see which samples
    mask_rv = bernoulli(0.5)
    # Since we are not using Mini-Batches computing the mask once suffices
    mask = tf.cast(mask_rv.rvs(size=(len(x_truth), n_heads, 1)), dtype=tf.float32)
    
    if len(np.shape(x_truth))==1:
        x_truth = np.expand_dims(x_truth, axis=-1)
    
    if len(np.shape(y_truth))==1:
        y_truth = np.expand_dims(y_truth, axis=-1)
    
    build_shape = list(np.shape(x_truth))[1:]
    print(np.shape(x_truth))
  
    #data_shape[0]=None
    
    model = bootstrap_model.BootstrapModel(dropout, n_heads, mask, tuple(build_shape))
 
    #model.build(tuple(data_shape))
    
    
    
    def loss_fcn(ground_truth, heads):
        labels = tf.cast(tf.tile(tf.expand_dims(ground_truth, axis=1), [1, n_heads, 1]), dtype=heads.dtype)
        loss = tf.nn.l2_loss(mask * (heads - labels))
        return loss
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    #@tf.function
    def train_step(inputs, ground_truth):
        with tf.GradientTape() as tape:
            heads, mean, variance = model(inputs)
            loss = loss_fcn(ground_truth, heads)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, heads, mean, variance


    for epoch in range(epochs):
        loss, heads, mean, variance = train_step(x_truth, y_truth)

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            print("Loss {}".format(loss))
            print("================")

    print("Training done!")
    return model


def bootstrap_training_old(x_truth, y_truth, dropout, learning_rate, epochs, n_heads, display_step=2000):
    """
    Generic training of Boostrap Network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param n_heads: Number of heads for trained Network
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder, mask_placeholder
    """
    tf.reset_default_graph()
    x_placeholder = tf.placeholder(tf.float32, [None, 1])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])
    dropout_placeholder = tf.placeholder(tf.float32)

    # This placeholder holds the mask indicating which heads see which samples
    mask_placeholder = tf.placeholder(tf.float32, shape=(None, n_heads, 1))

    heads, mean, variance = bootstrap_model.bootstrap_model(x_placeholder, dropout_placeholder,
                                                            n_heads, mask_placeholder)
    tf.add_to_collection('prediction', mean)
    tf.add_to_collection('uncertainties', variance)
    tf.add_to_collection('heads', heads)

    labels = tf.tile(tf.expand_dims(y_placeholder, axis=1), [1, n_heads, 1])
    # Loss is also only computed on masked heads
    loss = tf.nn.l2_loss(mask_placeholder * (heads - labels))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    mask_rv = bernoulli(0.5)
    # Since we are not using Mini-Batches computing the mask once suffices
    mask = mask_rv.rvs(size=(len(x_truth), n_heads, 1))
    
    for epoch in range(epochs):
        feed_dict = {x_placeholder: x_truth.reshape([-1, 1]),
                     y_placeholder: y_truth.reshape([-1, 1]),
                     dropout_placeholder: dropout,
                     mask_placeholder: mask}

        sess.run(train, feed_dict=feed_dict)

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            current_loss = sess.run(loss, feed_dict=feed_dict)
            print("Loss {}".format(current_loss))
            print("================")

    print("Training done")
    return sess, x_placeholder, dropout_placeholder, mask_placeholder


if __name__ == "__main__":
    #x, y = sample_generators.generate_osband_nonlinear_samples()
    x, y = sample_generators.generate_osband_sin_samples()
    sess = bootstrap_training(x, y, dropout=0.3, learning_rate=1e-3, epochs=6000, n_heads=5, display_step=200)
