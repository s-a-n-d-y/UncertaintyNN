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
pre_trained = False
# Seed defined for aligning images and their masks
seed = 1
batch_size = 9
num_epochs = 50
model_weights = "camvid_model_"+ str(num_epochs) +"_epochs.h5"
model_weights_checkpoint = "camvid_model_"+ str(num_epochs) +"_epochs_checkpoint.h5"
steps_per_epoch = np.ceil(float(len(camvid.frames_list) - round(0.1*len(camvid.frames_list))) / float(batch_size))
validation_steps = (float((round(0.1*len(camvid.frames_list)))) / float(batch_size))

print("Steps per epoch is: ", steps_per_epoch)
print("Validation steps per epoch is: ", validation_steps)

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

model = unet(n_filters = 32, batch_size = batch_size)
model.compile(optimizer='adam', 
              loss={
                  'segmentation_output': 'categorical_crossentropy',
                  'uncertainity_output': my_loss_fn
                   }, 
              metrics={
                  'segmentation_output': 'categorical_accuracy',
                  'uncertainity_output': 'mse'})
model.summary()

# Tensorboard settings
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath=model_weights_checkpoint, monitor='accuracy', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='val_accuracy', patience=10, verbose=1)
callbacks = [tb, mc, es]

if pre_trained:
    model.load_weights(model_weights)
else:
    result = model.fit(dataAugmentGenerator(img_dir, train_frames_datagen, train_masks_datagen, train_frames_folder, train_masks_folder, seed = seed, batch_size = batch_size), 
                    steps_per_epoch=steps_per_epoch,
                    batch_size = batch_size,
                    validation_data = dataAugmentGenerator(img_dir, val_frames_datagen, val_masks_datagen, val_frames_folder, val_masks_folder, seed = seed, batch_size = batch_size),
                    validation_steps = validation_steps, epochs=num_epochs, callbacks=callbacks)
    model.save_weights(model_weights, overwrite=True)

    ##########################################################################################################
    ##################################### Training statistics ################################################
    ##########################################################################################################
    # Get actual number of epochs model was trained for
    N = len(result.history['loss'])

    #Plot the model evaluation history
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20,8))

    fig.add_subplot(1,2,1)
    plt.title("Training Loss")
    plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
    plt.ylim(0, 1)

    fig.add_subplot(1,2,2)
    plt.title("Training Accuracy")
    plt.plot(np.arange(0, N), result.history["segmentation_output"], label="train_accuracy")
    plt.plot(np.arange(0, N), result.history["val_segmentation_output"], label="val_accuracy")
    plt.ylim(0, 1)

    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

##############################################################################################################
########################################## Inference Satistics ###############################################
##############################################################################################################

testing_gen = dataAugmentGenerator(img_dir, val_frames_datagen, val_masks_datagen, val_frames_folder, val_masks_folder, seed = seed, batch_size = batch_size)
batch_img,batch_mask = next(testing_gen)
pred_all= model.predict(batch_img)

#print(np.shape(pred_all))

for i in range(0,np.shape(pred_all)[0]):
    
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(batch_img[i])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(camvid.onehot_to_rgb(batch_mask[i], camvid.id2code))
    ax2.grid(b=None)
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Predicted labels')
    ax3.imshow(camvid.onehot_to_rgb(pred_all[i], camvid.id2code))
    ax3.grid(b=None)
    
    plt.show()