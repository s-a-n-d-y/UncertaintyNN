# Path imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data.Dataloader import *
from models.unet_model import *

# Normal pythonic imports
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image

# TF libraries
import tensorflow as tf

def dataAugmentGenerator(img_dir, frames_datagen, masks_datagen, frames_folder, masks_folder, seed = 1, batch_size = 5):
    '''Train Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    image_generator = frames_datagen.flow_from_directory(
    (img_dir + frames_folder),
    batch_size = batch_size, seed = seed)

    mask_generator = masks_datagen.flow_from_directory(
    (img_dir + masks_folder),
    batch_size = batch_size, seed = seed)

    while True:
        X1i = image_generator.next()
        X2i = mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [camvid.rgb_to_onehot(X2i[0][x,:,:,:], camvid.id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)


##############################################################################################################
############################# Sample Training code for plugging in later #####################################
##############################################################################################################

# Download data from here: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip
img_dir = '/home/sandy/Projects/PhD/Courses/WASP/DL-GAN/WASP-DL-GAN-HA/HA3/Group/CamSeq01/'
folder = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val']
camvid = Dataloader(img_dir, folder)
camvid.visualize_sample(n_images_to_show=2)
# Normalizing only frame images, since masks contain label info
data_gen_args = dict(rescale=1./255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)

train_frames_folder = folder[0].split('/')[0]
train_masks_folder = folder[1].split('/')[0]
val_frames_folder = folder[2].split('/')[0]
val_masks_folder = folder[3].split('/')[0]

##############################################################################################################
############################### Here we can load our uncertainity models #####################################
##############################################################################################################

# Seed defined for aligning images and their masks
seed = 1
model = unet(n_filters = 32)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['categorical_accuracy'])
model.summary()

# Tensorboard settings
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='camvid_model_150_epochs_checkpoint.h5', monitor='accuracy', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='val_accuracy', patience=10, verbose=1)
callbacks = [tb, mc, es]


batch_size = 5
num_epochs = 10
steps_per_epoch = np.ceil(float(len(camvid.frames_list) - round(0.1*len(camvid.frames_list))) / float(batch_size))
validation_steps = (float((round(0.1*len(camvid.frames_list)))) / float(batch_size))

print("Steps per epoch is: ", steps_per_epoch)
print("Validation steps per epoch is: ", validation_steps)

result = model.fit(dataAugmentGenerator(img_dir, train_frames_datagen, train_masks_datagen, train_frames_folder, train_masks_folder, seed = 1, batch_size = 5), 
                    steps_per_epoch=18,
                    validation_data = dataAugmentGenerator(img_dir, val_frames_datagen, val_masks_datagen, val_frames_folder, val_masks_folder, seed = 1, batch_size = 5),
                    validation_steps = validation_steps, epochs=num_epochs, callbacks=callbacks)
model.save_weights("camvid_model_150_epochs.h5", overwrite=True)



