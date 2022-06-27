# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import preprocessing_factory
import utils
import model
import time
import os
import re
from datetime import datetime
import math
# tf.app.flags.DEFINE_string("model_file", './fast-style-model.ckpt-30000', "")
# tf.app.flags.DEFINE_string("generated_image_file", './', "")
tf.app.flags.DEFINE_string("model_file", './zoe-generate/[base]IN-batch_size_type=4/Asheville_huang-c.1-s.250.0-r.100-tv.0.0001/fast-style-model.ckpt-26000', "")
tf.app.flags.DEFINE_string("generated_image_file", './zoe-generate/[base]IN-batch_size_type=4/Asheville_huang-c.1-s.250.0-r.100-tv.0.0001/', "")
tf.app.flags.DEFINE_string("image_file", "img/test.jpg", "")

TIMESTAMP="{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. ')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("generated_image_name", "res-" + TIMESTAMP + ".jpg", "")
tf.app.flags.DEFINE_list('style_strength', [0.0, 0.25, 0.50, 0.75, 1.0], '[0.0,1.0]')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # Get image's height and width.
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = utils.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)
            image = tf.tile(image, [len(FLAGS.style_strength), 1, 1, 1]) # 复制image

            generated = model.net(image, FLAGS.style_strength, training=False) # (1, 476, 712, 3)（H:474 W:712）   [有padding：shape=(1, 456, 692, 3) [0,255] float]
            generated = tf.cast(generated, tf.uint8) # shape=(1, 456, 692, 3) [0,255] int
            res = [tf.image.encode_jpeg(generated[i]) for i in range(len(FLAGS.style_strength))]

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            start_time = time.time()
            res_tmp = sess.run(res)
            tf.logging.info('Elapsed time/pic: %fs' % (time.time() - start_time))
            
            # Make sure 'generated' directory exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            for i in range(len(FLAGS.style_strength)):
                generated_file = FLAGS.generated_image_file + '[final][' + str(FLAGS.style_strength[i]) + ']' + FLAGS.generated_image_name #'generated/res.jpg'
                with open(generated_file, 'wb') as img:
                    img.write(res_tmp[i])                    
                    tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
