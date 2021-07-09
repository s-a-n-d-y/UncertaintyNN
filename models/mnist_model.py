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
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads
        
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
        
        heads = []
        for _ in range(n_heads):
            logits = tf.keras.layers.Dense(10)(x)
            class_prob = tf.nn.softmax(logits)
            heads.append([logits, class_prob])
            
        self.model = tf.keras.Model(inputs = inputs, outputs = heads)

        
    def call(self, x):
        if len(tf.shape(x))!=4:
            x = tf.reshape(x, [-1, 28, 28, 1])
            
        return self.model(x)


def bootstrap_cnn_mnist_model(x, dropout_rate, n_heads=5, reuse=False):
    """
    Last FC-Layer has n heads.
    """
    
    return BootstrapCNNMnistModel(dropout_rate, n_heads = n_heads)(x)
