import tensorflow as tf


class DropoutModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self.keep_prob = 1 - dropout_rate
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x):
        output = self.model(x)
        predictions = tf.reshape(output, [-1, 1])
        
        return predictions
        

def dropout_model(x, dropout_rate):
    """
    Constructs Dropout network to process simple 2D data.
    After every weight layer a dropout layer is placed.

    :param x: Input feature x
    :param dropout_rate:
    :return: prediction
    """
    return DropoutModel(dropout_rate)(x)
    