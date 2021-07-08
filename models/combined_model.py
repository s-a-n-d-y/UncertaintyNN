import tensorflow as tf


class CombinedModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        
        self.keep_prob = 1 - dropout_rate
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(2)
        ])
        
    def call(self, x):
        output = self.model(x)
        
        predictions = tf.expand_dims(output[:,0], -1)
        log_variance = tf.expand_dims(output[:,1], -1)
        
        return predictions, log_variance


def combined_model(x, dropout_rate):
    """
    Model that combines aleatoric and epistemic uncertainty.
    Based on the "What uncertainties do we need" paper by Kendall.
    Works for simple 2D data.

    :param x: Input feature x
    :param dropout_rate:
    :return: prediction, log(sigma^2)
    """

    model = CombinedModel(dropout_rate)
    return model(x)
    

