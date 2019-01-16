from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging 
import numpy as np
import tensorflow as tf

import data_input 
import wic_model
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in.')

# Model specific paramenters
flags.DEFINE_string('model_dir', '', 'Output model directory')
flags.DEFINE_integer('noise_dim', 1024, 'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 1024, 'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', 8, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 100, 'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 100, 'Steps per interior TPU loop. Should be less than  --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.0002, 'LR for both D and G')
flags.DEFINE_boolean('eval_loss', False, 'Evaluate discriminator and generator loss during eval')
flags.DEFINE_boolean('use_tpu', True, 'Use TPU for training')

_NUM_VIZ_IMAGES = 100 # For generating a 10x10 grid of generator samples

dataset = None
model = None

def model_fn(features, labels, mode, params):
    del labels
    if mode == tf.estimator.ModeKeys.PREDICT:
        random_noise = features['random_noise']
        predictions = {
                'generated_images': model.generator(random_noise, is_training=False)
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    batch_size = params['batch_size']
    real_images = features['real_images']
    random_noise = features['random_noise']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    generated_images = model.generator(random_noise, is_training=is_training)

    d_on_data_logits = tf.squeeze(model.discriminator(real_images))
    d_on_g_logits = tf.squeeze(model.discriminator(generated_images))

    # Calculate discriminator loss
    d_loss_on_data = tf.reduce_mean(tf.nn.relu(1. - d_on_data_logits))
    d_loss_on_gen = tf.reduce_mean(tf.nn.relu(1. + d_on_g_logits))

    d_loss = d_loss_on_data + d_loss_on_gen

    # Calculate generator loss
    g_loss = - tf.reduce_mean(d_on_g_logits)

    #Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        d_optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate, beta1=0.5)

        if FLAGS.use_tpu:
            d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
            g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_step = d_optimizer.minimize(
                    d_loss,
                    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator'))
            g_step = g_optimizer.minimize(
                    g_loss,
                    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Generator'))
            increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
            joint_op = tf.group([d_step]*5 + [g_step, increment_step])

            return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=g_loss,
                    train_op=joint_op)

    #Evaluation
    elif mode == tf.estimator.ModeKeys.EVAL:
        def _eval_metric_fn(d_loss, g_loss):
            return {
                    'discriminator_loss': tf.metrics.mean(d_loss),
                    'generator_loss': tf.metrics.mean(g_loss)}

        return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=g_loss,
                eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))

    # Should never reach here
    raise ValueError('Invalid mode provided to model_fn')


def generate_input_fn(is_training):
    return dataset.InputFunction(is_training, FLAGS.noise_dim)


def noise_input_fn(params):
    np.random.seed(0)
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
            np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    return {'random_noise': noise}, None


def main(argv):
    del argv
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)

    config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    num_shards=FLAGS.num_shards,
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    per_host_input_for_training=False))

    global dataset, model
    dataset = data_input
    model = wic_model

    # TPU-based estimator used for TRAIN and EVAL
    est = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            config=config,
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size)

    # CPU-based estimator used for PREDICT (generating images)
    cpu_est = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=False,
            config=config,
            predict_batch_size=_NUM_VIZ_IMAGES)

    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir) 
    tf.logging.info('Starting training for %d steps, current step: %d' % (FLAGS.train_steps, current_step))
    while current_step < FLAGS.train_steps:
        next_checkpoint = min(current_step + FLAGS.train_steps_per_eval, FLAGS.train_steps)
        est.train(input_fn=generate_input_fn(True), max_steps=int(next_checkpoint))
        current_step = next_checkpoint
        tf.logging.info('Finished training step %d' % current_step)

        if FLAGS.eval_loss:
            # Evaluate loss on test set
            metrics = est.evaluate(input_fn=generate_input_fn(False),
                                    steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
            tf.logging.info('Finished evaluating')
            tf.logging.info(metrics)

        # Render some generated images
        generated_iter = cpu_est.predict(input_fn=noise_input_fn)
        images = [p['generated_images'][:, :, :] for p in generated_iter]
        assert len(images) == _NUM_VIZ_IMAGES
        image_rows = [np.concatenate(images[i:i+10], axis=0) for i in range(0, _NUM_VIZ_IMAGES, 10)]
        tiled_image = np.concatenate(image_rows, axis=1)

        img = dataset.convert_array_to_image(tiled_image)

        step_string = str(current_step).zfill(5)
        file_obj = tf.gfile.Open(
                os.path.join(FLAGS.model_dir, 'generated_images', 'gen_%s.png' % (step_string)), 'w')
        img.save(file_obj, format='png')
        tf.logging.info('Finished generating images')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
