import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.applications import InceptionV3, VGG16, MobileNetV2, ResNet101, EfficientNetB1
from tqdm import tqdm

def gen_feat_with_model(model_name, train_images, test_images):
    if model_name == 'inception':
        train_images = tf.keras.applications.inception_v3.preprocess_input(train_images)
        test_images = tf.keras.applications.inception_v3.preprocess_input(test_images)
        train_images = tf.image.resize(train_images, (75, 75))
        test_images = tf.image.resize(test_images, (75, 75))
        # Min input size for inceptionnet is 75
        model = InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(75, 75, 3),
            pooling='max'
        )
    elif model_name == 'vgg':
        train_images = tf.keras.applications.vgg16.preprocess_input(train_images)
        test_images = tf.keras.applications.vgg16.preprocess_input(test_images)
        model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling='max'
        )
    elif model_name == 'mobilenet':
        train_images = tf.keras.applications.mobilenet_v2.preprocess_input(train_images)
        test_images = tf.keras.applications.mobilenet_v2.preprocess_input(test_images)
        model = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling='max'
        )
    elif model_name == 'resnet':
        train_images = tf.keras.applications.resnet.preprocess_input(train_images)
        test_images = tf.keras.applications.resnet.preprocess_input(test_images)
        model = ResNet101(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling='max'
        )
    elif model_name == 'efficientnet':
        train_images = tf.keras.applications.efficientnet.preprocess_input(train_images)
        test_images = tf.keras.applications.efficientnet.preprocess_input(test_images)
        model = EfficientNetB1(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling='max'
        )
    else:
        print('Invalid model name')
        return
    model.trainable = False

    train_results = model(train_images[:500])
    for i in tqdm(range(500, len(train_images), 500)):
        train_results = np.concatenate((train_results, model(train_images[i:i+500]).numpy()), axis=0)
    print(train_results.shape)
    np.save('./data/CIFAR100_%s_train_feat' % model_name, train_results)

    test_results = model(test_images[:500])
    for i in tqdm(range(500, len(test_images), 500)):
        test_results = np.concatenate((test_results, model(test_images[i:i+500]).numpy()), axis=0)
    print(test_results.shape)
    np.save('./data/CIFAR100_%s_test_feat' % model_name, test_results)

if __name__ == '__main__':
    print(f"gpu: { len(tf.config.list_physical_devices('GPU')) }")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    for model_name in ['efficientnet', 'inception', 'resnet', 'mobilenet', 'vgg']:
        print("Starting with model %s" % model_name)
        gen_feat_with_model(model_name, train_images, test_images)
        print("Done with model %s" % model_name)
