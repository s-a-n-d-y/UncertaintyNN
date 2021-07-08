import tensorflow as tf


class DropoutCNNMnistModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=[5,5], padding="SAME", activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
            tf.keras.layers.Conv2D(64, kernel_size=[5,5], padding="SAME", activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10)
        ])
        
    def call(self, x):
        if len(tf.shape(x))!=4:
            x = tf.reshape(x, [-1, 28, 28, 1])
        logits = self.model(x)
        return logits, tf.nn.softmax(logits)


def dropout_cnn_mnist_model(x, dropout_rate, reuse=False):
    """
    Builds a simple CNN MNIST classifier with dropout after every layer
    that contains learned weights.

    :param x:
    :param reuse: True if reusing layer scopes
    :return:
    """
    
    return DropoutCNNMnistModel(dropout_rate)(x)



class CombinedCNNMnistModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        inputs = tf.keras.Input((28,28,1))
        x = inputs 
        
        x = tf.keras.layers.Conv2D(32, kernel_size=[5,5], padding="SAME", activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2)(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=[5,5], padding="SAME", activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        y = tf.keras.layers.Dense(10)(x)
        z = tf.keras.layers.Dense(10)(x)
        
        self.model = tf.keras.Model(inputs = inputs, outputs = [y,z])
        
    def call(self, x):
        if len(tf.shape(x))!=4:
            x = tf.reshape(x, [-1, 28, 28, 1])
            
        logits, uncertainty = self.model(x)
        return logits, tf.nn.softmax(logits), uncertainty
        


def combined_cnn_mnist_model(x, dropout_rate, reuse=False):
    """
    Model learns to predict aleatoric uncertainty as output
    """
    
    return CombinedCNNMnistModel(dropout_rate)(x)


class BootstrapCNNMnistModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, n_heads = 5):
        super().__init_()
        
        


import numpy as np
spoof = np.random.random((10,28,28))

print(CombinedCNNMnistModel(0.4)(spoof))


def bootstrap_cnn_mnist_model(x, dropout_rate, n_heads=5, reuse=False):
    """
    Last FC-Layer has n heads.
    """
    input_layer = tf.reshape(x, [-1, 28, 28, 1])

    heads = []
    with tf.variable_scope("mnist_model"):
        # Convolutional Layer #1
        with tf.variable_scope("conv1", reuse=reuse):
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            dropout1 = tf.layers.dropout(conv1, dropout_rate, training=True)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        with tf.variable_scope("conv2", reuse=reuse):
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(conv2, dropout_rate, training=True)

        pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2)

        # Dense Layer
        with tf.variable_scope("fc1", reuse=reuse):
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout3 = tf.layers.dropout(dense, dropout_rate, training=True)

        # Logits Layer + HEAD layer
        with tf.variable_scope("fc2", reuse=reuse):
            for i in range(n_heads):
                logits = tf.layers.dense(inputs=dropout3, units=10)
                class_prob = tf.nn.softmax(logits, name="softmax_tensor")

                heads.append([logits, class_prob])
                # heads.append({
                #     "logits": logits,
                #     "class_prob:" class_prob
                # })


    return heads
