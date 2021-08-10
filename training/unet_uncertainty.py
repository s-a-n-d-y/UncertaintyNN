import sys
sys.path.append("../")
import numpy as np
from models.unet_model import Unet
from models.unet_model import CombinedLoss
import tensorflow as tf

from tensorflow.keras.datasets import mnist

import matplotlib
#matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from scipy import misc

import os

from PIL import Image
import pdb

actual_classes = np.array(
        [
            [64, 128, 64],
            [192, 0, 128],
            [0, 128, 192],
            [0, 128, 64],
            [128, 0, 0],
            [64, 0, 128],
            [64, 0, 192],
            [192,128,64],
            [192,192,128],
            [64,64,128],
            [128,0,192],
            [192,0,64],
            [128,128,64],
            [192,0,192],
            [128,64,64],
            [64,192,128],
            [64,64,0],
            [128,64,128],
            [128,128,192],
            [0,0,192],
            [192,128,128],
            [128,128,128],
            [64,128,192],
            [0,0,64],
            [0,64,64],
            [192,64,128],
            [128,128,0],
            [192,128,192],
            [64,0,64],
            [192,192,0],
            [0,0,0],
            [64,192,0]
        ]
    )   


def extract_img_number(file_list):
    arr = []
    for name in file_list:
        arr.append(int(name[7:12]))
    return np.array(arr)


def extract_file_id(file_list):
    arr = []
    for name in file_list:
        arr.append("_".join(name[:-4].split("_")[:2]))
    return arr


def class_to_rgb(data):

    dict_tuple_to_class = {}
    dict_class_to_tuple = {}
    tuple_classes = [tuple(x) for x in actual_classes]
    for ind, tup in enumerate(tuple_classes):
        dict_tuple_to_class[tup]=ind
        dict_class_to_tuple[ind]=tup

    data_shape = np.shape(data)
    output_shape = data_shape + (3,)
    output = np.zeros(output_shape)
    for bb in range(data_shape[0]):
        for ii in range(data_shape[1]):
            for jj in range(data_shape[2]):
                pixel_class = int(data[bb,ii,jj])
                output[bb,ii,jj] = list(dict_class_to_tuple[pixel_class])

    return np.array(output)


def rgb_to_class(data):

    dict_tuple_to_class = {}
    dict_class_to_tuple = {}
    tuple_classes = [tuple(x) for x in actual_classes]
    for ind, tup in enumerate(tuple_classes):
        dict_tuple_to_class[tup]=ind
        dict_class_to_tuple[ind]=tup

    data_shape = np.shape(data)
    output_shape = data_shape[:-1]
    output = np.zeros(output_shape)
    for bb in range(data_shape[0]):
        for ii in range(data_shape[1]):
            for jj in range(data_shape[2]):
                pixel_rgb = tuple(data[bb,ii,jj])
                try:
                    output[bb,ii,jj] = dict_tuple_to_class[pixel_rgb]
                except:
                    output[bb,ii,jj] = -1

    return np.array(output)


def get_data(data_type="train", image_shape=(256,256), data_limit=None, shuffled=False):
    # 3frames_path = f"/home/oscar/Work/Studies/WASP/GANs/Module_3_assignment/UncertaintyNN/data/{data_type}_frames/{data_type}"
    # masks_path = f"/home/oscar/Work/Studies/WASP/GANs/Module_3_assignment/UncertaintyNN/data/{data_type}_masks/{data_type}"
    frames_path = f"../data/camvid/{data_type}/images"
    masks_path = f"../data/camvid/{data_type}/labels"

    tmp_frame_files = os.listdir(frames_path)
    tmp_mask_files = os.listdir(masks_path)

    # frame_ids = extract_img_number(tmp_frame_files)
    # mask_ids = extract_img_number(tmp_mask_files)
    frame_ids = extract_file_id(tmp_frame_files)
    mask_ids = extract_file_id(tmp_mask_files)

    frame_files = []
    mask_files = []

    used_frames = []
    used_masks = []

    for number, name in zip(frame_ids, tmp_frame_files):
        if number in mask_ids:
            frame_files.append(name)
            used_frames.append(number)

    frame_files = np.array(frame_files)

    for number, name in zip(mask_ids, tmp_mask_files):
        if number in frame_ids:
            mask_files.append(name)
            used_masks.append(number)

    mask_files = np.array(mask_files)

    frame_dict = {}
    mask_dict = {}
    for name in frame_files:
        seq = name.split("_")[0]
        try:
            frame_dict[seq]
        except KeyError:
            frame_dict[seq] = []

        frame_dict[seq].append(name)

    for name in mask_files:
        seq = name.split("_")[0]
        try:
            mask_dict[seq]
        except KeyError:
            mask_dict[seq] = []

        mask_dict[seq].append(name)

    frame_files = []
    mask_files = []
    for key in frame_dict:
        frame_files.append(np.sort(frame_dict[key]))
        mask_files.append(np.sort(mask_dict[key]))

    frame_files = np.concatenate(frame_files, axis=0)
    mask_files = np.concatenate(mask_files, axis=0)

    # frame_files = frame_files[np.argsort(used_frames)]
    # mask_files = mask_files[np.argsort(used_masks)]

    if shuffled:
        shuffled_inds = np.arange(len(frame_files))
        np.random.shuffle(shuffled_inds)
        frame_files = np.squeeze(frame_files[shuffled_inds])
        mask_files = np.squeeze(mask_files[shuffled_inds])

    frames = []
    for ind, frame in enumerate(frame_files):
        image_path = os.path.join(frames_path, frame)
        image_data = Image.open(image_path)
        reshaped_image = image_data.resize(image_shape, Image.NEAREST)
        data = np.array(reshaped_image)
        frames.append(data)
        if isinstance(data_limit, int) and ind >= data_limit:
            break

    frames = np.array(frames)

    masks = []
    for ind, mask in enumerate(mask_files):
        image_path = os.path.join(masks_path, mask)
        image_data = Image.open(image_path)
        reshaped_image = image_data.resize(image_shape, Image.NEAREST)
        data = np.array(reshaped_image)
        masks.append(data)
        if isinstance(data_limit, int) and ind >= data_limit:
            break

    masks = np.array(masks) 

    masks = rgb_to_class(masks)

    # print(np.shape(masks))

    # print(np.unique(masks))

    # plt.imshow(masks[0])
    # plt.show()

    # diffed_masks = []

    # for class_rgb in actual_classes:
    #     diffed_masks.append(np.linalg.norm(masks - class_rgb, axis=-1))

    # diffed_masks = np.array(diffed_masks)

    # masks = np.argmin(diffed_masks, axis=0)

    return frames, masks

"""
Animal 64 128 64
Archway 192 0 128
Bicyclist 0 128 192
Bridge 0 128 64
Building 128 0 0
Car 64 0 128
CartLuggagePram 64 0 192
Child 192 128 64
Column_Pole 192 192 128
Fence 64 64 128
LaneMkgsDriv 128 0 192
LaneMkgsNonDriv 192 0 64
Misc_Text 128 128 64
MotorcycleScooter 192 0 192
OtherMoving 128 64 64
ParkingBlock 64 192 128
Pedestrian 64 64 0
Road 128 64 128
RoadShoulder 128 128 192
Sidewalk 0 0 192
SignSymbol 192 128 128
Sky 128 128 128
SUVPickupTruck 64 128 192
TrafficCone 0 0 64
TrafficLight 0 64 64
Train 192 64 128
Tree 128 128 0
Truck_Bus 192 128 192
Tunnel 64 0 64
VegetationMisc 192 192 0
Void 0 0 0
Wall 64 192 0
"""
if __name__=="__main__":
    print("Loading data...", end=" ")
    train_frames, train_masks = get_data("train", shuffled=True)
    val_frames, val_masks = get_data("val", shuffled=True)
    test_frames, test_masks = get_data("test", shuffled=True)
    print("Done!")

    # shuffled_train_ids = np.random.shuffle(np.arange(len(train_frames)))
    # train_frames = np.squeeze(train_frames[shuffled_train_ids])
    # train_masks = np.squeeze(train_masks[shuffled_train_ids])

    # shuffled_val_ids = np.random.shuffle(np.arange(len(val_frames)))
    # val_frames = np.squeeze(val_frames[shuffled_val_ids])
    # val_masks = np.squeeze(val_masks[shuffled_val_ids])

    # shuffled_test_ids = np.random.shuffle(np.arange(len(test_frames)))
    # test_frames = np.squeeze(test_frames[shuffled_test_ids])
    # test_masks = np.squeeze(test_masks[shuffled_test_ids])

    limited_data = 80

    fraction = 1/3.5

    train_x = train_frames[:int(len(train_frames)*fraction)] / 255
    train_y = train_masks[:int(len(train_masks)*fraction)]

    # val_split = 0.3

    val_x = val_frames[:int(len(val_frames)*fraction)] / 255  # train_x[:int(np.ceil(val_split*len(train_x)))]
    val_y = val_masks[:int(len(val_masks)*fraction)]  # train_y[:int(np.ceil(val_split*len(train_y)))]

    # train_x = train_x[int(np.ceil(val_split*len(train_x))):]
    # train_y = train_y[int(np.ceil(val_split*len(train_y))):]

    test_x = test_frames[:int(len(test_frames)*fraction)] / 255
    test_y = test_masks[:int(len(test_masks)*fraction)]

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=32)
    val_y = tf.keras.utils.to_categorical(val_y, num_classes=32)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(5)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(5)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(5)


    # (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # limited_data = 20
    #     
    # 
    # train_x = train_x[:limited_data]
    # train_y = train_y[:limited_data]
    # test_x = test_x[:limited_data]
    # test_y = test_y[:limited_data]
    # 
    # goal_size = np.array([64,64])
    # mnist_size = np.shape(train_x)[1:]
    # difference = goal_size - mnist_size
    # 
    # top_pad = difference[0]//2
    # bottom_pad = difference[0]//2 + difference[0]%2
    # 
    # row_pad = (top_pad, bottom_pad)
    # 
    # left_pad = difference[1]//2
    # right_pad = difference[1]//2 + difference[1]%2
    # 
    # col_pad = (left_pad, right_pad)
    # 
    # batch_pad = (0,0)
    # 
    # train_x_padded = resize(train_x, (limited_data,64,64))
    # test_x_padded = resize(test_x, (limited_data,64,64))
    # 
    # # train_x_padded = np.pad(train_x, (batch_pad, row_pad, col_pad))
    # # test_x_padded = np.pad(test_x, (batch_pad, row_pad, col_pad))
    # 
    # threshold = 100/255
    # train_masks = [[[cls + 1 if cell > threshold else 0 for cell in col] for col in image] for image, cls in zip(train_x_padded, train_y)]
    # 
    # test_masks = [[[cls + 1 if cell > threshold else 0 for cell in col] for col in image] for image, cls in zip(test_x_padded, test_y)]
    # 
    # train_x_padded = np.expand_dims(train_x_padded, axis=-1) / 255
    # train_masks_categorical = tf.keras.utils.to_categorical(train_masks, num_classes=11)
    # 
    # val_x = train_x_padded[:int(np.ceil(0.2*limited_data))]
    # val_masks = train_masks_categorical[:int(np.ceil(0.2*limited_data)),:,:]
    # 
    # train_x_padded = train_x_padded[int(np.ceil(0.2*limited_data)):]
    # train_masks_categorical = train_masks_categorical[int(np.ceil(0.2*limited_data)):]
    # 
    # test_x_padded = np.expand_dims(test_x_padded, axis=-1) / 255
    # test_masks_categorical = tf.keras.utils.to_categorical(test_masks, num_classes=11)
    # 
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_x_padded, train_masks_categorical)).batch(100)

    # spoof = np.random.random((10,64,64,3))
    # spoof_labels = np.random.random((10,64,64,32))

    print("Setting up model...")
    batch_shape = np.shape(train_x)[1:]
    output_classes = np.shape(train_y)[-1]
    model = Unet(num_blocks=5, batch_shape=batch_shape, output_classes=output_classes, num_var_outputs=1)
    model.build(input_shape=(None,) + batch_shape)
    model.summary()
    loss_fcn = CombinedLoss(output_classes=output_classes)
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    iou = tf.keras.metrics.MeanIoU(num_classes=output_classes)

    # class_probs, log_var = model(spoof)
    # # print(loss_fcn(spoof_labels, class_probs))
    # # print(np.shape(class_probs))
    # # print(np.shape(log_var))
     
    def train_epoch(data, labels):
        with tf.GradientTape() as tape:
            class_probs, log_var = model(data)
            loss = loss_fcn(labels, class_probs)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss, class_probs, log_var

    epochs = 100
    epoch_log_variances = []
    nans = False
    best_val_acc = 0
    for epoch in range(epochs):
        print(f"Start epoch {epoch}/{epochs-1}")
        step_log_variances = []
        for step, (img_batch, lab_batch) in enumerate(train_dataset):
            model.set_training()
            loss, train_probs, log_var = train_epoch(img_batch, lab_batch)
            if np.isnan(loss).any() or np.isnan(train_probs).any() or np.isnan(log_var).any():
                nans = True
                break
            train_probs = np.mean(train_probs, axis=0)
            train_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(lab_batch, train_probs))

            step_log_variances.append(log_var)
            print(f"Step {step}: \t Loss = {loss} \t Acc = {train_acc}")



            # diff_along_sample = np.diff(train_probs, axis=0)
            # print("Max difference between sample predictions", np.max(np.abs(diff_along_sample)))

        # np.save(f"train_frames_step_{step}", img_batch)
        # np.save(f"train_masks_step_{step}", lab_batch)
        # np.save(f"train_probs_step_{step}", train_probs)
        # np.save(f"train_log_var_step_{step}", log_var)

        epoch_log_variances.append(step_log_variances)

        model.set_evaluation()
        probs = []
        labs = []
        for step, (img_batch, lab_batch) in enumerate(val_dataset):
            val_probs, _ = model(img_batch)

            #preds = tf.argmax(val_probs, axis=-1)
            #iou.update_state(np.argmax(lab_batch, axis=-1), preds)

            probs.append(val_probs)
            labs.append(lab_batch)

        val_masks = tf.concat(labs, axis=0)
        val_probs = tf.concat(probs, axis=0)
        val_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(val_masks, val_probs))

        #val_iou = iou.result()
        #iou.reset_state()
            
        #print(f"Validation acc: {val_acc}\t, Validation IoU: {val_iou}")
        print(f"Validation acc: {val_acc}")
        if val_acc > best_val_acc:
            print("Best val acc thus far, saving checkpoint.")

            model.save_weights('./checkpoints/unet_with_uncertainty')
            best_val_acc = val_acc

        # if (epoch + 1) % 10:
        #     learning_rate = learning_rate * 0.90
        #     optimizer.learning_rate = learning_rate

        if nans:
            print("Breaking due to NaNs.")
            break



    # print(np.shape(epoch_log_variances))

    # while True:
    #     choice = input("What epoch, step, and sample do you want to view the variance from? (ints separated by ',' 'n' to cancel)\n")
    #     print(choice)
    #     choice = choice.replace(" ","")
    #     choice = choice.split(",")
    #     print(choice)
    #     if choice=="n" or choice=="N":
    #         break
    #     if len(choice)==1:
    #         epoch = int(choice[0])
    #         step = 0
    #         sample = 0
    #     elif len(choice)==2:
    #         epoch = int(choice[0])
    #         step = int(choice[0])
    #         sample = 0
    #     elif len(choice)==3:
    #         epoch = int(choice[0])
    #         step = int(choice[0])
    #         sample = int(choice[0])
    #     else:
    #         raise RuntimeError()
    # 
    #     print(epoch, type(epoch))
    #     print(step, type(step))
    #     print(sample, type(sample))
    #     plt.imshow(epoch_log_variances[epoch, step, sample])
    #     plt.show()

    model.set_evaluation()
    probs = []
    labs = []
    variances = []
    for step, (img_batch, lab_batch) in enumerate(test_dataset):
        test_probs, test_var = model(img_batch)
        probs.append(test_probs)
        labs.append(lab_batch)
        variances.append(test_var)
    test_masks = tf.concat(labs, axis=0)
    test_probs = tf.concat(probs, axis=0)
    test_var = tf.concat(variances, axis=0) 
    test_preds = np.argmax(test_probs, axis=-1)


    probs = []
    labs = []
    variances = []
    for step, (img_batch, lab_batch) in enumerate(train_dataset):
        train_probs, train_var = model(img_batch)
        probs.append(train_probs)
        labs.append(lab_batch)
        variances.append(train_var)
    train_masks = tf.concat(labs, axis=0)
    train_probs = tf.concat(probs, axis=0)
    train_var = tf.concat(variances, axis=0) 
    train_preds = np.argmax(train_probs, axis=-1)

    _, ax = plt.subplots(3,1 + 11 + 1)
    ax[0,0].imshow(train_x[0])
    ax[1,0].imshow(test_x[0])
    ax[2,0].imshow(test_x[1])
    if len(np.shape(train_var))==4:
        ax[0,-1].imshow(train_var[0,...,0])  # I just changed the number of variance outputs to 1 so some of these need to change
        ax[1,-1].imshow(test_var[0,...,0])
        ax[2,-1].imshow(test_var[1,...,0])
    else:
        ax[0,-1].imshow(train_var[0])  # I just changed the number of variance outputs to 1 so some of these need to change
        ax[1,-1].imshow(test_var[0])
        ax[2,-1].imshow(test_var[1])

    for ii in range(1,1 + 11):
        ax[0,ii].imshow(train_probs[0,:,:,ii-1])
        ax[1,ii].imshow(test_probs[0,:,:,ii-1])
        ax[2,ii].imshow(test_probs[1,:,:,ii-1])
    plt.show()

    _, ax = plt.subplots(5,1 + 11 + 1)
    for ii in range(5):
        ax[ii,0].imshow(train_x[ii])
        if len(np.shape(train_var))==4:
            ax[ii,-1].imshow(train_var[ii,...,0])
        else:
            ax[ii,-1].imshow(train_var[0])

    for ii in range(5):
        for jj in range(1,1 + 11):
            ax[ii,jj].imshow(train_probs[ii,:,:,jj-1])
    plt.show()

    num_plots = min(10, len(test_preds))

    if np.shape(test_var)[-1]==output_classes:
        var_preds = np.zeros_like(test_preds)   

        for bb in range(np.shape(test_var)[0]):
            for ii in range(np.shape(test_var)[1]):
                for jj in range(np.shape(test_var)[2]):
                    var_preds[bb,ii,jj] = test_var[bb,ii,jj,test_preds[bb,ii,jj]]
    else:
        var_preds = test_var

    _, ax = plt.subplots(4,num_plots)
    test_y = np.argmax(test_y, axis=-1)
    for ii in range(num_plots):
        ax[0,ii].imshow(test_x[ii])
        ax[1,ii].imshow(test_y[ii])
        ax[2,ii].imshow(test_preds[ii])
        ax[3,ii].imshow(var_preds[ii])

    plt.show()
