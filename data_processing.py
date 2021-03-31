from PIL import Image
import numpy as np
import os
import tensorflow as tf

imgs = []
labels = []
depths_single = []
depths_multi = []
depths_repeated = []
masks_single = []
masks_multi = []
masks_repeated = []

def normalize(arr):
    return arr / 255

for filename in os.listdir('./LFSD/focus_stack/'):
    img = Image.open('./LFSD/focus_stack/' + filename).resize((256, 256))
    arr = np.array(img)
    arr = normalize(arr)
    imgs.append(arr)
    labels.append(int(filename.split('_')[0]) - 1)

for i in range(1, 101):
    mask_single = normalize(np.array(Image.open('./LFSD/ground_truth/%d.png' % i).convert('L').resize((256, 256))))
    mask_single = np.expand_dims(mask_single, axis=2)
    masks_single.append(mask_single)
    

    depth_repeat = normalize(np.array(Image.open('./LFSD/depth_map/%d.bmp'%i).resize((256, 256))))
    depth_repeat = np.expand_dims(depth_repeat, axis=2)
    depth_repeat = np.repeat(depth_repeat, 3, axis=2)
    depths_repeated.append(depth_repeat)

    

np.save('./data/LFSD_imgs',np.array(imgs))
np.save('./data/LFSD_labels',np.array(labels))
np.save('./data/LFSD_depths_repeated',np.array(depths_repeated)[labels])
np.save('./data/LFSD_masks_single',np.array(masks_single)[labels])