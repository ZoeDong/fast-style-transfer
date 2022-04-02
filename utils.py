# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import yaml
import os
from os import listdir
from os.path import isfile, join
import nets_factory
import preprocessing_factory

slim = tf.contrib.slim

'''
reader
'''
def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)

def image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess_fn(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)


'''
losses
'''
def gram(layer):
    ''' 
    layer 在 conv1_2,conv2_2,conv3_3,conv4_3 上计算风格损失
    eg.Tensor("vgg_16/conv1/conv1_2/Relu:0", shape=(1, 256, 256, 64), dtype=float32) 
    num_images = 1 ; width = shape = 256 ; height = 256 ; num_filters = 64
    '''
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    ''' height、weight 相乘，reshape: Tensor("Reshape:0", shape=(1, 65536, 64), dtype=float32)'''
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    ''' gram shape: Tensor("truediv_21:0", shape=(1, 64, 64), dtype=float32) '''
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams

def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        """ 返回网络结构函数 """
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        """ 返回预处理/不进行预处理的函数 """
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)

        """ 获得style image """
        # Get the style image data
        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        # image = _aspect_preserving_resize(image, size)

        """ 扩展维度 3-D to 4-D 形成batch """
        # Add the batch dimension
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        # images = tf.stack([image_preprocessing_fn(image, size, size)])
        
        """
        神经网络函数处理图像,返回 神经网络全连接层Tensor 和 网络节点dict:
        net: Tensor("vgg_16/fc8/BiasAdd:0", shape=(1, 2, 2, 1), dtype=float32), 
        endpoints_dict: OrderedDict([('vgg_16/conv1/conv1_1', <tf.Tensor 'vgg_16/conv1/conv1_1/Relu:0' shape=(1, 256, 256, 64) dtype=float32>),.....]
        """
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        """ 依次计算 style layer 的 gram matrix 值"""
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)

        with tf.Session() as sess:
            # Restore variables for loss network.
            """ 剔除fc层的网络 """
            init_func = _get_init_fn(FLAGS)
            init_func(sess)

            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)

def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        # generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        # size = tf.size(generated_images)
        # layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        generated_images_1, generated_images_0, _, content_images_0 = tf.split(endpoints_dict[layer], 4, 0)
        size = tf.size(generated_images_1)
        layer_style_loss_1 = tf.nn.l2_loss(gram(generated_images_1) - style_gram) * 2 / tf.to_float(size)
        layer_style_loss_0 = tf.nn.l2_loss(gram(generated_images_0) - gram(content_images_0)) * 2 / tf.to_float(size)
        layer_style_loss = (layer_style_loss_1 + layer_style_loss_0)/2

        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary

def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images) # (4, 64, 64, 256) = 4194304
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss


'''
losses
'''
def _get_init_fn(FLAGS):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    """ 获得网络中不需要使用的层的名字,fc """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    """ variables_to_restore 只保留卷积层，不保留全连接层 """
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)

class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS

def mean_image_subtraction(image, means):
    image = tf.to_float(image)

    num_channels = 3
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)

if __name__ == '__main__':
    f = read_conf_file('conf/mosaic.yml')
    print(f.loss_model_file)
