import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class Unet(tf.keras.Model):
    def __init__(self, n_filters=16, dilation_rate=1, dropout_rate=0.3, output_classes=32, batch_size=5, train_mc_samples=5, batch_shape=(256,256,3), num_blocks=5, num_var_outputs=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.output_classes = output_classes
        self.train_mc_samples = train_mc_samples
        if num_var_outputs is None:
            self.num_var_outputs = self.output_classes
        else:
            self.num_var_outputs = 1
        self.training = True

        #Define input batch shape
        inputs = Input(batch_shape=(batch_size,) + batch_shape)
        skip_connections = []
        x = inputs
        for ii in range(num_blocks):

            x = Conv2D(n_filters * 2**ii, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            x = Conv2D(n_filters * 2**ii, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            skip_connections.append(x)
            x = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(x)

        for ii in np.flip(np.arange(num_blocks)):
            x = concatenate([UpSampling2D(size=(2, 2))(x), skip_connections[ii]], axis=3)
            x = Conv2D(n_filters * 2**ii, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            x = Conv2D(n_filters * 2**ii, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        #outputs = Conv2D(self.output_classes + 1, (1, 1), padding = 'same', dilation_rate = dilation_rate)(x)
        outputs = Conv2D(self.output_classes + self.num_var_outputs, (1, 1), padding = 'same', dilation_rate = dilation_rate)(x)

        self.model = Model(inputs=inputs, outputs=outputs)


    def summary(self):
        super().summary()
        layers = self.get_layers()
        for layer in layers:
            print("===========================================")
            print(f"Name: {layer.name}\t\t Output shape: {layer.output_shape}")
            print("Fed by: ", end=" ")
            int_node = layer._inbound_nodes[0]
            # print(int_node.inbound_layers)
            # out_node = layer._outbound_nodes[0]
            # print(out_node)
            predecessor_layers = int_node.inbound_layers
            if isinstance(predecessor_layers, list):
                names = []
                for pre_layer in predecessor_layers:
                    names.append(pre_layer.name)
                    # print(pre_layer.name, end="")
                print(names)
            else:
                print(predecessor_layers.name)


    def set_dropout_rate(self, dropout_rate):
        layers = self.get_layers()
        for layer in layers:
            if "dropout" in layer.name:
                layer.rate = dropout_rate


    def get_layers(self):
        return self.model.layers


    def set_evaluation(self):
        self.training = False


    def set_training(self):
        self.training = True


    def call(self, x):
        outputs = self.model(x)
        class_logits = outputs[...,:self.output_classes]
        #log_variance = outputs[...,-1]
        log_variance = outputs[...,self.output_classes:]

        variance = tf.exp(log_variance)  # To convert log_variance to variance
        std = tf.math.sqrt(variance)  # To convert variance to std
        if np.shape(std)[-1]!=self.output_classes:
            scaling = tf.tile(std, (1, 1, 1, self.output_classes))

        if self.training:
            class_probs = []
            for ii in range(self.train_mc_samples):
                noise = tf.random.normal(shape=tf.shape(class_logits))
                scaled_noise = noise * scaling 

                class_probs.append(tf.nn.softmax(class_logits + scaled_noise))
        else:
            noise = tf.random.normal(shape=tf.shape(class_logits))
            scaled_noise = noise * scaling

            class_probs = tf.nn.softmax(class_logits + scaled_noise)

        return class_probs, std


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, output_classes=32):
        super().__init__()
        self.output_classes = output_classes

    def call(self, y_true, y_pred):
        losses = []
        loss_fcn = tf.keras.losses.CategoricalCrossentropy()
        for mc_sample in y_pred:
            losses.append(loss_fcn(y_true, mc_sample))

        loss = tf.reduce_mean(losses)
        variance_loss = tf.math.log(
                    1/len(y_pred) * tf.reduce_sum(
                        tf.math.exp(
                            y_pred - tf.tile(tf.expand_dims(tf.math.log(tf.reduce_sum(tf.math.exp(y_pred), axis=-1)), axis=-1), (1, 1, 1, 1, self.output_classes))
                        ),
                        axis=0
                    )
                )
            
        loss = variance_loss + loss

        return loss





def unet(n_filters = 16, batch_size=5, dilation_rate = 1):
    '''Validation Image data generator
        Inputs: 
            n_filters - base convolution filters
            dilation_rate - convolution dilation rate
        Output: Unet keras Model
    '''
    #Define input batch shape
    inputs = Input(batch_shape=(batch_size, 256, 256, 3))
    
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
    conv1 = BatchNormalization()(conv1) 
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
    conv3 = BatchNormalization()(conv3)  
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
    conv4 = BatchNormalization()(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
    conv5 = BatchNormalization()(conv5)   
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
    conv6 = BatchNormalization()(conv6)   
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
    conv6 = BatchNormalization()(conv6)
        
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
    conv7 = BatchNormalization()(conv7)  
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
    conv7 = BatchNormalization()(conv7)
        
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
    conv8 = BatchNormalization()(conv8)  
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
    conv8 = BatchNormalization()(conv8)
        
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
    conv9 = BatchNormalization()(conv9)   
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
    conv9 = BatchNormalization()(conv9)
        
    conv10 = Conv2D(32, (1, 1), activation='softmax', padding = 'same', dilation_rate = dilation_rate)(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    
    return model
