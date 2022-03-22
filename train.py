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
import time

TIMESTAMP="{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

slim = tf.contrib.slim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/denoised_starry.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    """ 计算图像风格特征 """
    time1=time.time()
    style_features_t = utils.get_style_features(FLAGS)
    # print(style_features_t[0].shape) # (64, 64)
    # print(style_features_t[1].shape) # (128, 128)
    # print(style_features_t[2].shape) # (256, 256)
    # print(style_features_t[3].shape) # (512, 512)
    # print(style_features_t,'\n')

    time2=time.time()
    print("********** spend time = ",time2-time1,'\tstyle_features_t')
    
    # Make sure the training path exists.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming + '-' + TIMESTAMP)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)
    time3=time.time()
    print("********** spend time = ",time3-time2,'\ttraining path')
    

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            """ 返回网络结构函数 """
            time4=time.time()
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False)
            print("********** spend time = ",time4-time3,'\tBuild Network')

            """ 返回预处理/不进行预处理的函数 """
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            
            """ 以batch读取数据集图片 """
            processed_images = utils.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)
            time5=time.time()
            print("********** spend time = ",time5-time4,'\tread image')
            
            
            """ 内容图片 -> 生成图片 """
            style_strength_list = [0.1*x for x in range(11)]
            style_strength = style_strength_list[np.random.randint(0,11)]
            # print("*********** before image.shape:",processed_images.shape)
            # image_v = tf.pad(processed_images, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
            # print("*********** after  image.shape:",image_v.shape)
            generated = model.net(processed_images, style_strength, training=True)
            time6=time.time()
            print("********** spend time = ",time6-time5,'\tgenerate image')

            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)
            time7=time.time()
            print("********** spend time = ",time7-time6,'\tprocessed_generated')

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
            time8=time.time()
            print("********** spend time = ",time8-time7,'\tcount loss')

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
            time9=time.time()
            print("********** spend time = ",time9-time8,'\tsummary')

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)


            # print("************* tf.trainable_variables:",tf.trainable_variables())
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
            # Restore variables for loss network.
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            """ 从checkpoint中加载变量 """
            # Restore variables for training model if the checkpoint file exists.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)
            time10=time.time()
            print("********** spend time = ",time10-time9,'\tprepare to train')

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
                    # print(step)

                    # for v in range(5):
                    #     name = sess.graph.get_tensor_by_name('res' + str(v+1) + '/residual/Variable:0')
                    #     print('res', str(v+1), '/residual/Variable:0 = ', sess.run(name))

                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                        tf.logging.info('content loss: %f,  original_style_loss: %f' % (content_loss.eval(), style_loss.eval()))
                        # print(type(processed_images)) # <class 'tensorflow.python.framework.ops.Tensor'>
                        # print(processed_images.shape) # (4, 256, 256, 3)
                        # print(processed_images) # Tensor("batch:0", shape=(4, 256, 256, 3), dtype=float32)
                        # print('\n')

                        # print(type(generated)) # <class 'tensorflow.python.framework.ops.Tensor'>
                        # print(generated.shape) # (4, 256, 256, 3)
                        # print(generated) # Tensor("Slice_1:0", shape=(4, 256, 256, 3), dtype=float32)
                        # print('\n')

                        # print(type(processed_generated)) # <class 'tensorflow.python.framework.ops.Tensor'>
                        # print(processed_generated.shape) # (4, 256, 256, 3)
                        # print(processed_generated) # Tensor("stack_11:0", shape=(4, 256, 256, 3), dtype=float32)
                        # print('\n')

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
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)
