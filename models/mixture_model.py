import tensorflow as tf
#import tensorflow.contrib.layers as layers


class MixtureModel(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, n_mixtures):
        super().__init__()
        
        self.sigma_max = 5
        self.keep_prob = 1 - dropout_rate

        self.n_mixtures = n_mixtures
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(50, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.keep_prob),
            tf.keras.layers.Dense(self.n_mixtures * 3)
        ])
        
    def call(self, x):
        output_layer = self.model(x)
        
        raw_output = tf.reshape(output_layer, [-1, self.n_mixtures, 3])
        mixture_weights = tf.nn.softmax(raw_output[:, :, 0] - tf.expand_dims(tf.reduce_max(raw_output[:, :, 0],
                                                                                           axis=1), 1))
        mixture_means = raw_output[:, :, 1]
        mixture_variances = self.sigma_max * tf.sigmoid(raw_output[:, :, 2])

        # Stacking along axis=1 might be easier
        gmm = tf.stack([mixture_weights, mixture_means, mixture_variances])

        mean = tf.reduce_sum(mixture_weights * mixture_means, axis=1)
        aleatoric_uncertainty = tf.reduce_sum(mixture_weights * mixture_variances, axis=1)
        epistemic_uncertainty = tf.reduce_sum(mixture_weights *
                                              tf.square(mixture_means - tf.expand_dims(
                                                  tf.reduce_sum(mixture_weights * mixture_means, axis=1),
                                                  axis=1
                                              )), axis=1)

        uncertainties = tf.stack([aleatoric_uncertainty, epistemic_uncertainty])
        
        return gmm, mean, uncertainties


def mixture_model(x, dropout_rate, n_mixtures):
    """
    Constructs a Mixture Density Network to process simple 2D data

    :param x: Input feature x
    :param dropout_rate:
    :param n_mixtures: Number of mixtures
    :return: gmm, mean, uncertainties
    """
    
    return MixtureModel(dropout_rate, n_mixtures)(x)

