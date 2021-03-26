import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.applications import InceptionV3
from tqdm import tqdm
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()


################################ RUN ONCE AND NEVER AGAIN #####################################
# train_images = tf.keras.applications.inception_v3.preprocess_input(train_images)
# test_images = tf.keras.applications.inception_v3.preprocess_input(test_images)
# train_images = tf.image.resize(train_images, (75, 75))
# test_images = tf.image.resize(test_images, (75, 75))
# print('Finished Processing')
# np.save('./data/CIFAR100_inception_train', train_images)
# np.save('./data/CIFAR100_inception_test', test_images)
################################ RUN ONCE AND NEVER AGAIN #####################################

train_images = np.load('./data/CIFAR100_inception_train.npy')
test_images = np.load('./data/CIFAR100_inception_test.npy')

inception = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(75, 75, 3),
    pooling='max'
)
inception.trainable = False

train_results = inception(train_images[:500])
for i in tqdm(range(500, len(train_images), 500)):
    train_results = np.concatenate((train_results, inception(train_images[i:i+500]).numpy()), axis=0)
print(train_results.shape)
np.save('./data/CIFAR100_inception_train_feat', train_results)

test_results = inception(test_images[:500])
for i in tqdm(range(500, len(test_images), 500)):
    test_results = np.concatenate((test_results, inception(test_images[i:i+500]).numpy()), axis=0)
print(test_results.shape)
np.save('./data/CIFAR100_inception_test_feat', test_results)