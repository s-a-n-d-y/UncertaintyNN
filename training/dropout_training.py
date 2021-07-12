import sys
sys.path.append('../')

from models import dropout_model
from data import sample_generators

import tensorflow as tf
import numpy as np


def dropout_training(x_truth, y_truth, dropout, learning_rate, epochs, display_step=2000):
    """
    Generic training of a Dropout Network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder
    """
    model = dropout_model.DropoutModel(dropout)
    
    if len(np.shape(x_truth))==1:
        x_truth = np.expand_dims(x_truth, axis=-1)
        
    if len(np.shape(y_truth))==1:
        y_truth = np.expand_dims(y_truth, axis=-1)
    
    data_shape = list(np.shape(x_truth))
    data_shape[0]=None
    model.build(tuple(data_shape))
    
    def loss_fcn(ground_truth, prediction):
        return tf.keras.losses.MeanSquaredError()(ground_truth, prediction)

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    #@tf.function
    def train_step(inputs, ground_truth):
        with tf.GradientTape() as tape:
            prediction = model(inputs)
            loss = loss_fcn(ground_truth, prediction)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, prediction
    
    for epoch in range(epochs):
        loss, prediction = train_step(x_truth, y_truth)
        
        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            print("Loss {}".format(loss))
            print("================")
            
    print("Training done!")
    
    return model


def dropout_training_old(x_truth, y_truth, dropout, learning_rate, epochs, display_step=2000):
    """
    Generic training of a Dropout Network for 2D data.

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

    prediction = dropout_model.dropout_model(x_placeholder, dropout_placeholder)

    tf.add_to_collection("prediction", prediction)

    loss = tf.losses.mean_squared_error(y_placeholder, prediction)

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

