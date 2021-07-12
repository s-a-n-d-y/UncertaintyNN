import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image
from pylab import *
import os
import sys
import tensorflow as tf

# Adapted from here: https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb

class Dataloader:

    def __init__(self, img_dir, folder):
        self.img_dir = img_dir
        self.folders = folder
        label_codes, label_names = zip(*[self.parse_code(l) for l in open(img_dir+"label_colors.txt")])
        self.label_codes, self.label_names = list(label_codes), list(label_names)
        self.code2id = {v:k for k,v in enumerate(self.label_codes)}
        self.id2code = {k:v for k,v in enumerate(self.label_codes)}
        self.create_folders()
        self.frame_tensors, self.masks_tensors, self.frames_list, self.masks_list = self.read_images(self.img_dir)
        self.create_training_data(self.frame_tensors, self.masks_tensors, self.frames_list, self.masks_list)
        

    def create_folders(self):
        for folder in self.folders:
            try:
                os.makedirs(self.img_dir + folder)
            except Exception as e: print(e)

        

    def read_to_tensor(self, fname, output_height=256, output_width=256, normalize_data=False):
        '''Function to read images from given image file path, and provide resized images as tensors
            Inputs: 
                fname - image file path
                output_height - required output image height
                output_width - required output image width
                normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
            Output: Processed image tensors
        '''
        
        # Read the image as a tensor
        img_strings = tf.io.read_file(fname)
        imgs_decoded = tf.image.decode_jpeg(img_strings)
        
        # Resize the image
        output = tf.image.resize(imgs_decoded, [output_height, output_width])
        
        # Normalize if required
        if normalize_data:
            output = (output - 128) / 128
        return output

    def read_images(self, img_dir):
        '''Function to get all image directories, read images and masks in separate tensors
            Inputs: 
                img_dir - file directory
            Outputs 
                frame_tensors, masks_tensors, frame files list, mask files list
        '''
        
        # Get the file names list from provided directory
        file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        
        # Separate frame and mask files lists, exclude unnecessary files
        frames_list = [file for file in file_list if ('_L' not in file) and ('txt' not in file)]
        masks_list = [file for file in file_list if ('_L' in file) and ('txt' not in file)]
        
        print('{} frame files found in the provided directory.'.format(len(frames_list)))
        print('{} mask files found in the provided directory.'.format(len(masks_list)))
        
        # Create file paths from file names
        frames_paths = [os.path.join(img_dir, fname) for fname in frames_list]
        masks_paths = [os.path.join(img_dir, fname) for fname in masks_list]
        
        # Create dataset of tensors
        frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
        masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)
        
        # Read images into the tensor dataset
        frame_tensors = frame_data.map(self.read_to_tensor)
        masks_tensors = masks_data.map(self.read_to_tensor)
        
        print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
        print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))
        
        return frame_tensors, masks_tensors, frames_list, masks_list


    def parse_code(self, l):
        '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
        '''
        if len(l.strip().split("\t")) == 2:
            a, b = l.strip().split("\t")
            return tuple(int(i) for i in a.split(' ')), b
        else:
            a, b, c = l.strip().split("\t")
            return tuple(int(i) for i in a.split(' ')), c

    def rgb_to_onehot(self, rgb_image, colormap):
        '''Function to one hot encode RGB mask labels
            Inputs: 
                rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
                colormap - dictionary of color to label id
            Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
        '''
        num_classes = len(colormap)
        shape = rgb_image.shape[:2]+(num_classes,)
        encoded_image = np.zeros( shape, dtype=np.int8 )
        for i, cls in enumerate(colormap):
            encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
        return encoded_image


    def onehot_to_rgb(self, onehot, colormap):
        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros( onehot.shape[:2]+(3,) )
        for k in colormap.keys():
            output[single_layer==k] = colormap[k]
        return np.uint8(output)


    def create_training_data(self, frames, masks, frames_list, masks_list):
        '''Function to save images in the appropriate folder directories 
            Inputs: 
                frames - frame tensor dataset
                masks - mask tensor dataset
                frames_list - frame file paths
                masks_list - mask file paths
        '''
        #Create iterators for frames and masks
        frame_batches = tf.compat.v1.data.make_one_shot_iterator(frames)  # outside of TF Eager, we would use make_one_shot_iterator
        mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)
        
        #Iterate over the train images while saving the frames and masks in appropriate folders
        dir_name='train'
        for file in zip(frames_list[:-round(0.2*len(frames_list))],masks_list[:-round(0.2*len(masks_list))]):
            
            
            #Convert tensors to numpy arrays
            frame = frame_batches.next().numpy().astype(np.uint8)
            mask = mask_batches.next().numpy().astype(np.uint8)
            
            #Convert numpy arrays to images
            frame = Image.fromarray(frame)
            mask = Image.fromarray(mask)
            
            #Save frames and masks to correct directories
            frame.save(self.img_dir+'{}_frames/{}'.format(dir_name, dir_name)+'/'+file[0])
            mask.save(self.img_dir+'{}_masks/{}'.format(dir_name, dir_name)+'/'+file[1])
        
        #Iterate over the val images while saving the frames and masks in appropriate folders
        dir_name='val'
        for file in zip(frames_list[-round(0.2*len(frames_list)):],masks_list[-round(0.2*len(masks_list)):]):
            
            
            #Convert tensors to numpy arrays
            frame = frame_batches.next().numpy().astype(np.uint8)
            mask = mask_batches.next().numpy().astype(np.uint8)
            
            #Convert numpy arrays to images
            frame = Image.fromarray(frame)
            mask = Image.fromarray(mask)
            
            #Save frames and masks to correct directories
            frame.save(self.img_dir+'{}_frames/{}'.format(dir_name, dir_name)+'/'+file[0])
            mask.save(self.img_dir+'{}_masks/{}'.format(dir_name, dir_name)+'/'+file[1])
        
        print("Saved {} frames to directory {}".format(len(frames_list),self.img_dir))
        print("Saved {} masks to directory {}".format(len(masks_list),self.img_dir))
        


    def visualize_sample(self, n_images_to_show=1):
        # Make an iterator to extract images from the tensor dataset
        frame_batches = tf.compat.v1.data.make_one_shot_iterator(self.frame_tensors)  # outside of TF Eager, we would use make_one_shot_iterator
        mask_batches = tf.compat.v1.data.make_one_shot_iterator(self.masks_tensors)

        n_images_to_show = n_images_to_show

        for i in range(n_images_to_show):
            
            # Get the next image from iterator
            frame = frame_batches.next().numpy().astype(np.uint8)
            mask = mask_batches.next().numpy().astype(np.uint8)
            
            #Plot the corresponding frames and masks
            fig = plt.figure()
            fig.add_subplot(1,2,1)
            plt.imshow(frame)
            fig.add_subplot(1,2,2)
            plt.imshow(mask)
            plt.show()



