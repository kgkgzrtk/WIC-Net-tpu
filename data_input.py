from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

NUM_EVAL_IMAGES = 2000
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data_path', '',
                                        'Path to weather training data.')
flags.DEFINE_string('test_data_path', '', 'Path to test data.')


def parser(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string),
            })
    image = tf.decode_raw(features['image'], tf.float64)
    label = tf.decode_raw(features['label'], tf.float64)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image.set_shape([3*128*128])
    image = tf.reshape(image, [128, 128, 3])
    label = tf.reshape(label, [6])
    return image, label


class InputFunction(object):

    def __init__(self, is_training, noise_dim):
        self.is_training = is_training
        self.noise_dim = noise_dim
        self.data_file = (FLAGS.train_data_path if is_training
                                            else FLAGS.test_data_path)

    def __call__(self, params):
        batch_size = params['batch_size']
        dataset = tf.data.TFRecordDataset([self.data_file])
        dataset = dataset.map(parser, num_parallel_calls=8)
        dataset = dataset.prefetch(4*batch_size).cache().repeat()
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(2)
        images, labels = dataset.make_one_shot_iterator().get_next()

        images = tf.reshape(images, [batch_size, 128, 128, 3])
        labels = tf.reshape(labels, [batch_size, 6])

        random_noise = tf.random_normal([batch_size, self.noise_dim])

        features = {
                'real_images': images,
                'random_noise': random_noise}
        #tf.summary.image('input_image', images)

        return features, labels


def convert_array_to_image(array):
    img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
    return img
