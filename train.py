# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import nets_factory
import preprocessing_factory
import model
import time
import utils
import os
import argparse
from datetime import datetime
import numpy as np

TIMESTAMP="{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

slim = tf.contrib.slim

## Basic configuration
tf.app.flags.DEFINE_string('mode', 'single', 'single/mixed.')

tf.app.flags.DEFINE_string('style_image', 'img/denoised_starry.jpg', 'targeted style image.')
tf.app.flags.DEFINE_string('naming', 'denoised_starry', 'the name of this model.')

tf.app.flags.DEFINE_string('style_image_1', 'img/mosaic.jpg', 'targeted style image.')
tf.app.flags.DEFINE_string('naming_1', 'mosaic', 'the name of this model.')
tf.app.flags.DEFINE_float('style_weight_1', 100, 'weight for style_1 features loss')

tf.app.flags.DEFINE_list('style_strength',  [0, 0, 1/3, 1/3, 2/3, 2/3, 1, 1], '') 
tf.app.flags.DEFINE_string('dataset_path', './train2014', 'the path to dataset')

## Weight of the loss
tf.app.flags.DEFINE_integer('content_weight', 1, 'weight for content features loss')
tf.app.flags.DEFINE_float('style_weight', 100, 'weight for style features loss')
tf.app.flags.DEFINE_float('tv_weight', 0.0001, 'weight for total variation loss')
tf.app.flags.DEFINE_integer('reconstruction_weight', 100, 'weight for total reconstruction loss')

## The size, the iter number to run
tf.app.flags.DEFINE_integer('image_size', 256, '')
tf.app.flags.DEFINE_integer('batch_size', 8, '')
tf.app.flags.DEFINE_integer('epoch', 1, '')

tf.app.flags.DEFINE_string('model_path', 'models', 'root path to save checkpoint and events file. The final path would be <model_path>/<naming>')

## Loss Network
tf.app.flags.DEFINE_list('content_layers', ['vgg_16/conv3/conv3_3'], 'use these layers for content loss')
tf.app.flags.DEFINE_list('style_layers', ['vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_2', 'vgg_16/conv3/conv3_3', 'vgg_16/conv4/conv4_3'], 'use these layers for style loss')
tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'loss network.')
tf.app.flags.DEFINE_string('loss_model_file', 'pretrained/vgg_16.ckpt', 'the path to the checkpoint.')

FLAGS = tf.app.flags.FLAGS
# FLAGS.naming = FLAGS.style_image.split('/')[-1].split('.')[0]

def main(FLAGS):
    """ 计算图像风格特征 and Make sure the training path exists"""
    if FLAGS.mode == 'single':
        training_path = os.path.join(FLAGS.model_path, 'single-' + FLAGS.naming + '-c.' + str(FLAGS.content_weight) + '-s.' + str(FLAGS.style_weight) + '-r.' + str(FLAGS.reconstruction_weight) + '-tv.' + str(FLAGS.tv_weight))
        style_features_t = utils.get_style_features_single(FLAGS)
    elif FLAGS.mode == 'mixed':
        training_path = os.path.join(FLAGS.model_path, 'mixed-' + FLAGS.naming + '-' + FLAGS.naming_1 + '-c.' + str(FLAGS.content_weight) + '-s1.' + str(FLAGS.style_weight) + '-s2.' + str(FLAGS.style_weight_1) + '-r.' + str(FLAGS.reconstruction_weight))
        style_features_t, style_features_t_1 = utils.get_style_features_mixed(FLAGS)

    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            """ 返回网络结构函数 """
            network_fn = nets_factory.get_network_fn(FLAGS.loss_model, num_classes=1, is_training=False)

            """ 返回预处理/不进行预处理的函数 """
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            
            """ 以batch读取数据集图片 """
            processed_images = utils.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            FLAGS.dataset_path, image_preprocessing_fn, epochs=FLAGS.epoch) # Tensor("batch:0", shape=(4, 256, 256, 3), dtype=float32)

            """ 内容图片 -> 生成图片 """
            generated = model.net(processed_images, FLAGS.style_strength, training=True) # Tensor("Slice_1:0", shape=(4, 256, 256, 3), dtype=float32)

            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)  # Tensor("stack_11:0", shape=(4, 256, 256, 3), dtype=float32)

            """ 生成图片+内容图片 输入vgg网络 """
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0))
            """Build Losses"""
            tv_loss = utils.total_variation_loss(generated)  # use the unprocessed image
            content_loss = utils.content_loss(endpoints_dict, FLAGS.content_layers)
            if FLAGS.mode == 'single':
                style_loss, style_loss_summary = utils.style_loss_single(endpoints_dict, FLAGS.style_layers, FLAGS.style_strength, style_features_t, FLAGS.style_weight)
                reconstruction_loss = utils.reconstruction_loss(processed_images, processed_generated)
                loss = style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss + FLAGS.reconstruction_weight * reconstruction_loss
            elif FLAGS.mode == 'mixed':
                style_loss, style_loss_summary = utils.style_loss_mixed(endpoints_dict, FLAGS.style_layers, FLAGS.style_strength, style_features_t, style_features_t_1, FLAGS.style_weight, FLAGS.style_weight_1)
                loss = style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss)
            tf.summary.scalar('weighted_losses/regularizer_loss', tv_loss)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * FLAGS.tv_weight)
            tf.summary.scalar('weighted_losses/weighted_reconstruction_loss', reconstruction_loss * FLAGS.reconstruction_weight)
            tf.summary.scalar('total_loss', loss)

            # for layer in FLAGS.style_layers:
            #     tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)

            """ 定义可保存的变量 """
            variables_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore, max_to_keep=100, write_version=tf.train.SaverDef.V1)

            """ 定义可训练的变量 只保留 transform network 部分"""
            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)


            """ 初始化变量 """
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            """ 初始化损失网络变量 """
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            """ 从checkpoint中加载变量 """
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            """Start Training"""
            """
            开始训练
            coord:开启协程
            coord.join:保证线程的完全运行即线程锁,保证线程池中的每个线程完成运行后,再开启下一个线程.
            threads:开启多线程,提高训练速度.
            """
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step, content_loss_tmp, style_loss_tmp, reconstruction_loss_tmp = sess.run([train_op, loss, global_step, content_loss, style_loss, reconstruction_loss])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f, content loss: %f, style_loss: %f, reconstruction_loss: %f ' \
                                % (step, loss_t, elapsed_time, content_loss_tmp, style_loss_tmp, FLAGS.reconstruction_weight * reconstruction_loss_tmp)) # all losses are weighted
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        if step / 1000 == 1:
                            saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
                        else:
                            saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step, write_meta_graph=False)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'), global_step=step, write_meta_graph=False)
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(FLAGS)
