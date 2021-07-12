import sys
sys.path.append("../")
import numpy as np
from models.unet_model import Unet
from models.unet_model import CombinedLoss
import tensorflow as tf

spoof = np.random.random((10,64,64,3))
spoof_labels = np.random.random((10,64,64,32))
model = Unet(num_blocks=5, batch_shape=(64,64,3))
model.build(input_shape=(10,64,64,3))
model.summary()
loss_fcn = CombinedLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
class_probs, log_var = model(spoof)
# print(loss_fcn(spoof_labels, class_probs))
# print(np.shape(class_probs))
# print(np.shape(log_var))

def train_epoch(data, labels):
    with tf.GradientTape() as tape:
        class_probs, log_var = model(data)
        loss = loss_fcn(labels, class_probs)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss

for epoch in range(2):
    print(f"start epoch {epoch}")
    loss = train_epoch(spoof, spoof_labels)
    print(f"Loss: {loss}")


