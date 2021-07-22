import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.unet_model import Unet
from training.unet_uncertainty import get_data
from training.unet_uncertainty import class_to_rgb 

train_frames, train_masks = get_data()
test_frames, test_masks = get_data("val")

test_x = test_frames / 255
test_y = tf.keras.utils.to_categorical(test_masks)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(5)

batch_shape = np.shape(train_frames)[1:]
output_classes = 32

model = Unet(num_blocks=5, batch_shape=batch_shape, output_classes=output_classes, num_var_outputs=1)
model.load_weights("../training/checkpoints/unet_with_uncertainty")
# model.summary()

model.set_evaluation()

probs = []
stds = []
labels = []
test_imgs = []
for (test_batch, test_mask) in test_dataset:
    prob, std = model(test_batch)
    probs.append(prob)
    stds.append(std)
    labels.append(test_mask)
    test_imgs.append(test_batch)

probs = tf.concat(probs, axis=0)
stds = tf.concat(stds, axis=0)
labels = tf.concat(labels, axis=0)
imgs = tf.concat(test_imgs, axis=0)

test_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, probs)).numpy()

print("Test accuracy: ", test_acc)

num_plots = min(8, len(imgs))
_, ax = plt.subplots(6,num_plots)
labels = np.argmax(labels, axis=-1)
rgb_labels = class_to_rgb(labels)
preds = np.argmax(probs, axis=-1)
rgb_preds = class_to_rgb(preds)
for ii in range(num_plots):
    ax[0,ii].imshow(imgs[ii])
    ax[1,ii].imshow(labels[ii])
    ax[2,ii].imshow(rgb_labels[ii]/255)
    ax[3,ii].imshow(preds[ii])
    ax[4,ii].imshow(rgb_preds[ii]/255)
    ax[5,ii].imshow(stds[ii])

plt.show()


# plt.imshow(imgs[0, 2:-3, 2:-3])
# plt.show()
# plt.imshow(rgb_labels[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(rgb_preds[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(stds[0, 2:-3, 2:-3])
# plt.show()


epistemic_probs = []
epistemic_stds = []
for (test_batch, test_mask) in test_dataset:
    prob, std = model(test_batch, epistemic=True)
    epistemic_probs.append(prob)
    epistemic_stds.append(std)

epistemic_probs = tf.concat(epistemic_probs, axis=0)
epistemic_stds = tf.concat(epistemic_stds, axis=0)


num_plots = min(8, len(imgs))
_, ax = plt.subplots(6,num_plots)
epistemic_preds = np.argmax(epistemic_probs, axis=-1)
epistemic_rgb_preds = class_to_rgb(epistemic_preds)
for ii in range(num_plots):
    ax[0,ii].imshow(imgs[ii])
    ax[1,ii].imshow(labels[ii])
    ax[2,ii].imshow(rgb_labels[ii]/255)
    ax[3,ii].imshow(epistemic_preds[ii])
    ax[4,ii].imshow(epistemic_rgb_preds[ii]/255)
    ax[5,ii].imshow(epistemic_stds[ii])

plt.show()

plt.imshow(imgs[0, 2:-3, 2:-3])
plt.show()
plt.imshow(rgb_labels[0, 2:-3, 2:-3] / 255)
plt.show()
plt.imshow(epistemic_rgb_preds[0, 2:-3, 2:-3] / 255)
plt.show()
plt.imshow(epistemic_stds[0, 2:-3, 2:-3])
plt.show()
