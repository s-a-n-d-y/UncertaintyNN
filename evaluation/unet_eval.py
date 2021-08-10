import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.unet_model import Unet
from training.unet_uncertainty import get_data
from training.unet_uncertainty import class_to_rgb 

def intersect_and_union(pred_label, label, num_classes):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
    Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = label
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, nan_to_num=None):
    """Calculate Intersection and Union (IoU)
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         float: Overall iou on all images.
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            results[i], gt_seg_maps[i], num_classes,
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    print(total_area_label)
    print(total_area_pred_label)
    print(total_area_intersect)
    print(total_area_union)

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    all_iou = total_area_intersect.sum() / total_area_union.sum()
    # acc = total_area_intersect / total_area_label
    # iou = total_area_intersect / total_area_union

    # Avoid divide by zero errors
    # Initiate out value to ones, ignore operation where denominator is zero
    acc = np.divide(total_area_intersect, total_area_label, out=np.ones((num_classes,)), where=total_area_label!=0)
    iou = np.divide(total_area_intersect, total_area_union, out=np.ones((num_classes,)), where=total_area_union!=0)

    if nan_to_num is not None:
        return (
            all_acc,
            np.nan_to_num(acc, nan=nan_to_num),
            all_iou,
            np.nan_to_num(iou, nan=nan_to_num),
        )

    return all_acc, acc, all_iou, iou

print("Loading train data...")
train_frames, train_masks = get_data(data_limit=10)

_, ax = plt.subplots(2,10)
for ii in range(10):
    ax[0,ii].imshow(train_frames[ii])
    ax[1,ii].imshow(train_masks[ii])
plt.savefig("train_imgs.png")

print("Loading test data...")
test_frames, test_masks = get_data("val", shuffled=True)

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

print("Out of ", probs.numpy().size, " elements there are ", probs.numpy().size - np.count_nonzero(probs), " which are zero")

log_probs = np.log(probs)
log_probs[np.isnan(log_probs)] = -1e5
log_probs[log_probs < -1e9] = -1e9
entropy = -np.einsum("bijc,bijc->bij", probs, log_probs)
mean_entropy = np.mean(entropy)

test_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, probs)).numpy()


num_plots = min(4, len(imgs))
fig, ax = plt.subplots(num_plots,4)
labels = np.argmax(labels, axis=-1)
rgb_labels = class_to_rgb(labels)
preds = np.argmax(probs, axis=-1)
rgb_preds = class_to_rgb(preds)

iou = tf.keras.metrics.MeanIoU(num_classes=output_classes)
iou.update_state(labels, preds)
test_iou = iou.result().numpy()
iou.reset_state()

iou.update_state(labels, labels)
debug_iou_s1 = iou.result().numpy()
iou.reset_state()

iou.update_state(labels[2], labels[3])
debug_iou = iou.result().numpy()
iou.reset_state()

print("Test accuracy: ", test_acc)
print("Test IoU: ", test_iou)
print("Mean test entropy: ", mean_entropy)
print("Debug IoU (should be 1): ", debug_iou_s1)
print("IoU using two nearby masks: ", debug_iou)

vmin = np.amin(stds[:num_plots])
vmax = np.amax(stds[:num_plots])

for ii in range(num_plots):
    ax[ii,0].imshow(imgs[ii])
    ax[ii,0].set_axis_off()
    # ax[ii,1].imshow(labels[ii])
    # ax[ii,1].set_axis_off()
    ax[ii,1].imshow(rgb_labels[ii]/255)
    ax[ii,1].set_axis_off()
    # ax[ii,3].imshow(preds[ii])
    # ax[ii,3].set_axis_off()
    ax[ii,2].imshow(rgb_preds[ii]/255)
    ax[ii,2].set_axis_off()
    im = ax[ii,3].imshow(stds[ii], vmin=vmin, vmax=vmax)
    ax[ii,3].set_axis_off()
    #ax[5,ii].colorbar()
    fig.colorbar(im, ax=ax[ii, 3])

plt.savefig("aleatoric.png")
# plt.show()
plt.close(fig)

for ii in range(num_plots):
    plt.figure()
    plt.imshow(imgs[ii])
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(f"input_img_{ii}.png")
    plt.close()

    plt.figure()
    plt.imshow(rgb_labels[ii]/255)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(f"label_img_{ii}.png")
    plt.close()

    plt.figure()
    plt.imshow(rgb_preds[ii]/255)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(f"pred_img_{ii}.png")
    plt.close()

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(stds[ii], vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    plt.savefig(f"std_img_{ii}.png")
    plt.close()


# plt.imshow(imgs[0, 2:-3, 2:-3])
# plt.show()
# plt.imshow(rgb_labels[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(rgb_preds[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(stds[0, 2:-3, 2:-3])
# plt.show()


# epistemic_probs = []
# epistemic_stds = []
# for (test_batch, test_mask) in test_dataset:
#     prob, std = model(test_batch, epistemic=True)
#     epistemic_probs.append(prob)
#     epistemic_stds.append(std)
#
# epistemic_probs = tf.concat(epistemic_probs, axis=0)
# epistemic_stds = tf.concat(epistemic_stds, axis=0)
#
#
# num_plots = min(8, len(imgs))
# fig, ax = plt.subplots(6,num_plots)
# epistemic_preds = np.argmax(epistemic_probs, axis=-1)
# epistemic_rgb_preds = class_to_rgb(epistemic_preds)
# for ii in range(num_plots):
#     ax[0,ii].imshow(imgs[ii])
#     ax[1,ii].imshow(labels[ii])
#     ax[2,ii].imshow(rgb_labels[ii]/255)
#     ax[3,ii].imshow(epistemic_preds[ii])
#     ax[4,ii].imshow(epistemic_rgb_preds[ii]/255)
#     im = ax[5,ii].imshow(epistemic_stds[ii])
#     #ax[5,ii].colorbar()
#     fig.colorbar(im, ax=ax[5,ii])
#
# plt.savefig("epistemic.png")
# # plt.show()
#
# plt.imshow(imgs[0, 2:-3, 2:-3])
# plt.show()
# plt.imshow(rgb_labels[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(epistemic_rgb_preds[0, 2:-3, 2:-3] / 255)
# plt.show()
# plt.imshow(epistemic_stds[0, 2:-3, 2:-3])
# plt.show()
