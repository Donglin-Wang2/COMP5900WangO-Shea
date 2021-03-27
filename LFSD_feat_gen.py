import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.applications import InceptionV3, VGG16, MobileNetV2, ResNet101
from tqdm import tqdm

def gen_feat_with_model(model_name, imgs, depths):
    if model_name == 'inception':
        depths = tf.keras.applications.inception_v3.preprocess_input(depths)
        imgs = tf.keras.applications.inception_v3.preprocess_input(imgs)
        # Min input size for inceptionnet is 75
        model = InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
            pooling='max'
        )
    elif model_name == 'vgg':
        depths = tf.keras.applications.vgg16.preprocess_input(depths)
        imgs = tf.keras.applications.vgg16.preprocess_input(imgs)
        model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
            pooling='max'
        )
    elif model_name == 'mobilenet':
        depths = tf.keras.applications.mobilenet_v2.preprocess_input(depths)
        imgs = tf.keras.applications.mobilenet_v2.preprocess_input(imgs)
        model = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
            pooling='max'
        )
    elif model_name == 'resnet':
        depths = tf.keras.applications.resnet.preprocess_input(depths)
        imgs = tf.keras.applications.resnet.preprocess_input(imgs)
        model = ResNet101(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
            pooling='max'
        )
    else:
        print('Invalid model name')
        return
    model.trainable = False

    print('Forwarding...')
    img_result = model(imgs).numpy()
    depth_result = model(depths).numpy()
    print("Img and depths feature shapes are:")
    print(img_result.shape, depth_result.shape)
    print("Saving...")
    np.save('./data/LFSD_imgs_inception_feat.npy', img_result)
    np.save('./data/LFSD_depths_repeated_inception_feat.npy', depth_result)
    print('Done')

if __name__ == '__main__':
    depths = np.load('./data/LFSD_depths_repeated.npy')
    imgs = np.load('./data/LFSD_imgs.npy')
    for model_name in ['inception', 'vgg', 'resnet', 'mobilenet']:
        gen_feat_with_model(model_name, imgs, depths)