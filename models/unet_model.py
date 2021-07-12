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
    def __init__(self, n_filters=16, dilation_rate=1, dropout_rate=0.3, output_classes=32, batch_size=5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.output_classes = output_classes

        #Define input batch shape
        batch_shape=(256,256,3)
        inputs = Input(batch_shape=(batch_size,) + batch_shape)
        print(inputs)
        
        conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
        conv1 = BatchNormalization()(conv1)

        conv1 = tf.keras.layers.Dropout(self.dropout_rate)(conv1)
            
        conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
        conv1 = BatchNormalization()(conv1)
        
        conv1 = tf.keras.layers.Dropout(self.dropout_rate)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
        conv2 = BatchNormalization()(conv2)
            
        conv2 = tf.keras.layers.Dropout(self.dropout_rate)(conv2)

        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
        conv2 = BatchNormalization()(conv2)
        
        conv2 = tf.keras.layers.Dropout(self.dropout_rate)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
        conv3 = BatchNormalization()(conv3)
            
        conv3 = tf.keras.layers.Dropout(self.dropout_rate)(conv3)

        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
        conv3 = BatchNormalization()(conv3)

        conv3 = tf.keras.layers.Dropout(self.dropout_rate)(conv3)
            
        pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
        conv4 = BatchNormalization()(conv4)

        conv4 = tf.keras.layers.Dropout(self.dropout_rate)(conv4)
            
        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
        conv4 = BatchNormalization()(conv4)

        conv4 = tf.keras.layers.Dropout(self.dropout_rate)(conv4)
            
        pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
        conv5 = BatchNormalization()(conv5)

        conv5 = tf.keras.layers.Dropout(self.dropout_rate)(conv5)
            
        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
        conv5 = BatchNormalization()(conv5)

        conv5 = tf.keras.layers.Dropout(self.dropout_rate)(conv5)
            
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        
        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
        conv6 = BatchNormalization()(conv6)

        conv6 = tf.keras.layers.Dropout(self.dropout_rate)(conv6)
            
        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
        conv6 = BatchNormalization()(conv6)

        conv6 = tf.keras.layers.Dropout(self.dropout_rate)(conv6)
            
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        
        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
        conv7 = BatchNormalization()(conv7)

        conv7 = tf.keras.layers.Dropout(self.dropout_rate)(conv7)
            
        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
        conv7 = BatchNormalization()(conv7)

        conv7 = tf.keras.layers.Dropout(self.dropout_rate)(conv7)
            
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        
        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
        conv8 = BatchNormalization()(conv8)

        conv8 = tf.keras.layers.Dropout(self.dropout_rate)(conv8)
            
        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
        conv8 = BatchNormalization()(conv8)

        conv8 = tf.keras.layers.Dropout(self.dropout_rate)(conv8)
            
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        
        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
        conv9 = BatchNormalization()(conv9)

        conv9 = tf.keras.layers.Dropout(self.dropout_rate)(conv9)
            
        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv9 = tf.keras.layers.Dropout(self.dropout_rate)(conv9)
            
        conv10 = Conv2D(self.output_classes + 1, (1, 1), padding = 'same', dilation_rate = dilation_rate)(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def call(self, x):
        outputs = self.model(x)
        class_logits = outputs[...,:self.output_classes]
        class_probs = tf.nn.softmax(class_logits)
        log_variance = outputs[...,-1]
        return class_probs, log_variance


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

    model = Model(inputs=inputs, outputs=conv10)
    
    return model
