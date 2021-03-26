import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3

# labels = np.load('./data/LFSD_labels.npy')
depths = np.load('./data/LFSD_depth_repeated.npy')
imgs = np.load('./data/LFSD_imgs.npy')
# masks = np.load('./data/LFSD_masks_single.npy')

inception = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling='max'
)

img_result = inception(imgs).numpy()
depth_result = inception(depths).numpy()
print(result.shape)
np.save('./data/LFSD_imgs_inception_feat.npy', img_result)
np.save('./data/LFSD_depths_repeated_inception_feat.npy', depth_result)
print('Done')