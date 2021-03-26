import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.applications import InceptionV3, VGG16, MobileNetV2, ResNet101

vgg = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling='max'
)
vgg.summary()
mobilenet = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling='max'
)
mobilenet.summary()
resnet = ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling='max'
)
resnet.summary()