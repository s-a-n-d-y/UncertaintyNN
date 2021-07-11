import sys
sys.path.append('../')

from models import mnist_model
# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import bernoulli


def mnist_dropout_evaluation(n_passes=50, dropout_rate=0.5, learning_rate=1e-4, epochs=10, display_epoch=1):
    """

    :param n_passes:
    :param dropout_rate:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return:
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_data = tf.cast(x_train, dtype=tf.float32)
    y_data = tf.cast(tf.keras.utils.to_categorical(y_train), dtype=tf.float32)
    
    x_test = tf.cast(x_test, dtype=tf.float32)
    y_test = tf.cast(tf.keras.utils.to_categorical(y_test), dtype=tf.float32)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(50)  
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)  

    model = mnist_model.DropoutCNNMnistModel(dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    num_batches = len(train_dataset)
    
    def train_step(input_data, ground_truth):
        with tf.GradientTape() as tape:
            logits, class_prob = model(input_data)
            correct_prediction = tf.equal(tf.argmax(class_prob, 1), tf.argmax(ground_truth, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=ground_truth, logits=logits)
            
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, accuracy

    for epoch in range(epochs):
        #batch = train_dataset[epoch%num_batches]

        model.set_dropout_rate(dropout_rate)
        accs = []
        for step, (train_image_batch, train_label_batch) in enumerate(
                    train_dataset
                ):
            _, acc = train_step(train_image_batch, train_label_batch)
            accs.append(acc)
            
        train_accuracy = tf.reduce_mean(accs)
        

        if epoch % display_epoch == 0:
            print("==================")
            print("Epoch {}".format(epoch))
            # cur_loss = sess.run(loss, feed_dict={x_data: batch[0],
            #                                      y_data: batch[1]})
            model.set_dropout_rate(0)
            probs = []
            labs = []
            for _, (train_image_batch, train_label_batch) in enumerate(
                    train_dataset
                ):
                _, class_prob = model(train_image_batch)
                probs.append(class_prob)
                labs.append(train_label_batch)
            class_prob = tf.concat(probs, axis=0)
            ground_truth = tf.concat(labs, axis=0)
            correct_prediction = tf.equal(tf.argmax(class_prob, axis=1), tf.argmax(ground_truth, axis=1))
            train_accuracy_zero_drop = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy with dropout_rate 0: {}".format(train_accuracy_zero_drop))
            print("Accuracy during training with droput_rate {}: {}".format(dropout_rate, train_accuracy))


    # Running forwards passes
    # MC Forward passes on test set
    model.set_dropout_rate(0.5)
    preds = []
    labels = []
    for _, (test_image_batch, test_label_batch) in enumerate(
                test_dataset
            ):
        # Use tile instead of repeat, because np.repeat flattens results
        x_batch_multipass = np.tile(test_image_batch, (n_passes,)+(len(np.shape(test_image_batch))-1)*(1,))

        _, pred_y_multipass = model(x_batch_multipass)

        pred_y_multipass = np.reshape(pred_y_multipass.numpy(),(-1, n_passes, 10), order="F")
        pred_y_mean = tf.reduce_mean(pred_y_multipass, axis=1)
        preds.append(pred_y_mean)
        labels.append(test_label_batch)

    pred_y_mean = tf.concat(preds, axis=0)
    ground_truth = tf.concat(labels, axis=0)

    correct_prediction = tf.equal(tf.argmax(pred_y_mean, 1), tf.argmax(ground_truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accurarcy using dropout_rate {}: {} ".format(0.5, accuracy))


    # pred_y_multipass = pred_y_multipass.reshape(-1, n_passes)

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # np.array([[e] * 10 for e in batch[0]]).reshape(-1, 500)


def mnist_dropout_evaluation_old(n_passes=50, dropout_rate=0.5, learning_rate=1e-4, epochs=10000, display_step=2000):
    """

    :param n_passes:
    :param dropout_rate:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return:
    """
    mnist = input_data.read_data_sets("../data/MNIST-data", one_hot=True)

    x_data = tf.placeholder(tf.float32, shape=[None, 784])
    y_data = tf.placeholder(tf.float32, shape=[None, 10])
    dropout_rate_data = tf.placeholder(tf.float32)

    logits, class_prob = mnist_model.dropout_cnn_mnist_model(x_data, dropout_rate_data)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
    correct_prediction = tf.equal(tf.argmax(class_prob, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        batch = mnist.train.next_batch(50)

        sess.run(train_step, feed_dict={x_data: batch[0], y_data: batch[1], dropout_rate_data: dropout_rate})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            # cur_loss = sess.run(loss, feed_dict={x_data: batch[0],
            #                                      y_data: batch[1]})
            train_accuracy = sess.run(accuracy, feed_dict={
                x_data: batch[0], y_data: batch[1], dropout_rate_data:0}) # no dropout on single accuracy
            print("Accuracy: {}".format(train_accuracy))


    # Running forwards passes
    # MC Forward passes on test set

    x_test = mnist.test.images
    y_test = mnist.test.labels
    
    for i in xrange(10):
        x_batch, y_batch = mnist.test.next_batch(100)
        # Use tile instead of repeat, because np.repeat flattens results
        x_batch_multipass = np.tile(x_batch, n_passes).reshape(-1, 784)


        pred_y_multipass = sess.run(class_prob, feed_dict={
            x_data: x_batch_multipass, dropout_rate_data: 0.5})

        pred_y_multipass = pred_y_multipass.reshape(-1, n_passes, 10)
        pred_y_mean = pred_y_multipass.mean(axis=1)

        acc = sess.run(accuracy, feed_dict={
            x_data: x_batch, y_data:y_batch, dropout_rate_data: 0.5})
        print("Accurarcy {}".format(acc))


    # pred_y_multipass = pred_y_multipass.reshape(-1, n_passes)

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # np.array([[e] * 10 for e in batch[0]]).reshape(-1, 500)

def mnist_bootstrap_evaluation(n_heads=5, dropout_rate=0.3, learning_rate=1e-4, epochs=20000, display_step=2000):
    mnist = input_data.read_data_sets("../data/MNIST-data", one_hot=True)

    x_data = tf.placeholder(tf.float32, shape=[None, 784])
    y_data = tf.placeholder(tf.float32, shape=[None, 10])
    dropout_rate_data = tf.placeholder(tf.float32)

    heads = mnist_model.bootstrap_cnn_mnist_model(x_data, dropout_rate_data)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    loss_per_head = []
    train_per_head = []
    accuracy_per_head = []
    for head in heads:
        logits, class_prob = head
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
        loss_per_head.append(loss)
        train_per_head.append(optimizer.minimize(loss))

        correct_prediction = tf.equal(tf.argmax(class_prob, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_per_head.append(accuracy) 

    rv = bernoulli(0.5)
    mask = rv.rvs(size=(n_heads, 50))

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mnist")[:16] 
    saver = tf.train.Saver(var_list=variable_list)

    for epoch in range(epochs):
        batch = mnist.train.next_batch(50)
        x, y = batch

        for i, train_step in enumerate(train_per_head):
            masked_x = x[mask[i] == 1, :]
            masked_y = y[mask[i] == 1, :]
            sess.run(train_step, feed_dict={x_data: x, y_data: y, dropout_rate_data: 0.5})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            for i, a in enumerate(accuracy_per_head):
                cur_acc = sess.run(a, feed_dict={x_data: x, y_data: y, dropout_rate_data: 0.5})
                print("Head {}, Accuracy: {}".format(i, cur_acc))

            saver_path = saver.save(sess, "mnist_boostrap.ckpt")


if __name__ == "__main__":
    mnist_dropout_evaluation(epochs=2)
    # mnist_bootstrap_evaluation(display_step=50)
