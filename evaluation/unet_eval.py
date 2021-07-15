import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np

from models.unet_model import Unet

from training.unet_uncertainty import get_data

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
for (test_batch, test_mask) in test_dataset:
    prob, std = model(test_batch)
    probs.append(prob)
    stds.append(std)
    labels.append(test_mask)

probs = tf.concat(probs, axis=0)
stds = tf.concat(stds, axis=0)
labels = tf.concat(labels, axis=0)

test_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, probs)).numpy()

print("Test accuracy: ", test_acc)
