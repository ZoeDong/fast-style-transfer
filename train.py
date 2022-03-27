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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/denoised_starry.yml', help='the path to the conf file')
    parser.add_argument('-s', '--style_weight', default=50, type=int, help='the value of style weight')
    parser.add_argument('-d', '--dataset_path', default='./train2014', help='the path to dataset')
    return parser.parse_args()


def main(FLAGS):
    """ 计算图像风格特征 """
    style_features_t = utils.get_style_features(FLAGS)
    # print(style_features_t[0].shape) # (64, 64)
    # print(style_features_t[1].shape) # (128, 128)
    # print(style_features_t[2].shape) # (256, 256)
    # print(style_features_t[3].shape) # (512, 512)
    # print(style_features_t,'\n')
    
    # Make sure the training path exists.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming + '-' + str(FLAGS.style_weight) + '-' + TIMESTAMP)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            """ 返回网络结构函数 """
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False)

            """ 返回预处理/不进行预处理的函数 """
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            
            """ 以batch读取数据集图片 """
            processed_images = utils.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            FLAGS.dataset_path, image_preprocessing_fn, epochs=FLAGS.epoch) # Tensor("batch:0", shape=(4, 256, 256, 3), dtype=float32)

            """ 内容图片 -> 生成图片 """
            # style_strength_list = [0.1*x for x in range(11)]
            # style_strength = style_strength_list[np.random.randint(0,11)]
            style_strength = 1

            generated = model.net(processed_images, style_strength, training=True) # Tensor("Slice_1:0", shape=(4, 256, 256, 3), dtype=float32)

            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)  # Tensor("stack_11:0", shape=(4, 256, 256, 3), dtype=float32)

            """ 生成图片+内容图片 输入vgg网络 """
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

            # Log the structure of loss network
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            for key in endpoints_dict:
                tf.logging.info(key)

            """Build Losses"""
            content_loss = utils.content_loss(endpoints_dict, FLAGS.content_layers)
            style_loss, style_loss_summary = utils.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
            tv_loss = utils.total_variation_loss(generated)  # use the unprocessed image
            
            _, processed_images_0 = tf.split(processed_images, 2, 0)
            _, processed_generated_0 = tf.split(processed_generated, 2, 0)
            # reconstruction_loss = 0.001 * FLAGS.style_weight * tf.norm(tf.abs(processed_images_0 - processed_generated_0), 1)
            reconstruction_loss =  tf.norm(tf.abs(processed_images_0 - processed_generated_0), 1)
            weighted_reconstruction_loss = 0.0001 * reconstruction_loss

            loss = style_strength * FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + \
                   FLAGS.tv_weight * tv_loss + weighted_reconstruction_loss

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('losses/content_loss', content_loss)
            tf.summary.scalar('losses/style_loss', style_loss)
            tf.summary.scalar('losses/regularizer_loss', tv_loss)

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * FLAGS.style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * FLAGS.tv_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('origin', tf.stack([
                image_unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)


            """ 定义可训练的变量 """
            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            """ 定义可保存的变量 """
            variables_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore, max_to_keep=100, write_version=tf.train.SaverDef.V1)

            """ 初始化变量 """
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            """ 初始化损失网络变量 """
            # Restore variables for loss network.
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            """ 从checkpoint中加载变量 """
            # Restore variables for training model if the checkpoint file exists.
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
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    # if step % 10 == 0:
                    if 1:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f, content loss: %f, style_loss: %f, weighted_style_loss: %f, reconstruction_loss: %f, weighted_reconstruction_loss: %f, res1/layer_strength: %f, res2/layer_strength: %f, res3/layer_strength: %f, res4/layer_strength: %f, res5/layer_strength: %f, ' \
                                        % (step, loss_t, elapsed_time, content_loss.eval(), style_loss.eval(), FLAGS.style_weight * style_loss.eval(), reconstruction_loss.eval(), weighted_reconstruction_loss.eval(), sess.run(tf.get_default_graph().get_tensor_by_name("res1/residual/Variable:0")), sess.run(tf.get_default_graph().get_tensor_by_name("res2/residual/Variable:0")), sess.run(tf.get_default_graph().get_tensor_by_name("res3/residual/Variable:0")), sess.run(tf.get_default_graph().get_tensor_by_name("res4/residual/Variable:0")), sess.run(tf.get_default_graph().get_tensor_by_name("res5/residual/Variable:0"))))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'), global_step=step)
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    FLAGS.dataset_path = args.dataset_path
    FLAGS.style_weight = args.style_weight
    main(FLAGS)
