import tensorflow as tf


def mask_gradients(x, mask):
    """
    Helper function to propagate gradients only from masked heads.

    :param x: Tensor to be masked
    :param mask: Mask to select heads
    :return:
    """
    mask_h = tf.abs(mask - 1)
    return tf.stop_gradient(mask_h * x) + mask * x


class BootstrapModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, n_heads, mask, input_shape):
        super().__init__()
        self.keep_prob = 1 - dropout_rate
        self.n_heads = n_heads
        self.mask = mask
        
        inputs = tf.keras.Input(input_shape)
        heads = []
        for ii in range(n_heads):
            x = inputs
            x = tf.keras.layers.Dense(50, activation=tf.keras.activations.relu)(x)
            x = tf.keras.layers.Dropout(self.keep_prob)(x)
            x = tf.keras.layers.Dense(50, activation=tf.keras.activations.relu)(x)
            x = tf.keras.layers.Dense(1)(x)
            heads.append(x)
        self.model = tf.keras.Model(inputs=inputs, outputs = heads)
        
        
    def call(self, x):
        outputs = self.model(x)
        outputs = tf.stack(outputs, axis=1)
        outputs = mask_gradients(outputs, self.mask)
        
        mean, var = tf.nn.moments(outputs, axes=1)
        return outputs, mean, var

        
def bootstrap_model(x, dropout_rate, n_heads, mask):
    
    """
    Constructs model with n_heads bootstraps heads to process
    simple 2D data.

    :param x: input features x
    :param dropout_rate:
    :param n_heads: number of heads to use for bootstrapping
    :param mask: Mask to which heads are used for which samples

    :return: masked_heads, heads, mean, variance
    """
    
    input_shape = tf.shape(x)[1:]
    
    Bmodel = BootstrapModel(dropout_rate, n_heads, mask, input_shape)
    
    return Bmodel(x)
    




