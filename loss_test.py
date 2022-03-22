from __future__ import print_function
from __future__ import division
import tensorflow as tf
import nets_factory
import preprocessing_factory
import model
import utils
import os
import argparse
from datetime import datetime
import numpy as np
import time


slim = tf.contrib.slim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/denoised_starry.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    """ 计算图像风格特征 """
    style_features_t = utils.get_style_features(FLAGS)

    with tf.Graph().as_default():
        with tf.Session() as sess:
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
                                            'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)            
            
            """ 内容图片 -> 生成图片 """
            style_strength_list = [0.1*x for x in range(11)]
            style_strength = style_strength_list[np.random.randint(0,11)]
            generated = model.net(processed_images, style_strength, training=True)

            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)

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

            loss = style_strength * FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

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
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)

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
            _, loss_t, step = sess.run([train_op, loss, global_step])

            """logging"""
            if step % 10 == 0:
                tf.logging.info('step: %d,  total Loss %f' % (step, loss_t))
                tf.logging.info('content loss: %f,  original_style_loss: %f' % (content_loss.eval(), style_loss.eval()))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)
