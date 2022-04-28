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

def get_style_features_single(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        """ 返回网络结构函数 """
        network_fn = nets_factory.get_network_fn(FLAGS.loss_model, num_classes=1, is_training=False)
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
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        
        """
        神经网络函数处理图像,返回 神经网络全连接层Tensor 和 网络节点dict:
        net: Tensor("vgg_16/fc8/BiasAdd:0", shape=(1, 2, 2, 1), dtype=float32), 
        endpoints_dict: OrderedDict([('vgg_16/conv1/conv1_1', <tf.Tensor 'vgg_16/conv1/conv1_1/Relu:0' shape=(1, 256, 256, 64) dtype=float32>),.....]
        """
        _, endpoints_dict = network_fn(images)
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
            return sess.run(features)

def get_style_features_mixed(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        """ 返回网络结构函数 """
        network_fn = nets_factory.get_network_fn(FLAGS.loss_model, num_classes=1, is_training=False)
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
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)

        img_bytes_1 = tf.read_file(FLAGS.style_image_1)
        if FLAGS.style_image_1.lower().endswith('png'):
            image_1 = tf.image.decode_png(img_bytes_1)
        else:
            image_1 = tf.image.decode_jpeg(img_bytes_1)
        images_1 = tf.expand_dims(image_preprocessing_fn(image_1, size, size), 0)
        
        """
        神经网络函数处理图像,返回 神经网络全连接层Tensor 和 网络节点dict:
        net: Tensor("vgg_16/fc8/BiasAdd:0", shape=(1, 2, 2, 1), dtype=float32), 
        endpoints_dict: OrderedDict([('vgg_16/conv1/conv1_1', <tf.Tensor 'vgg_16/conv1/conv1_1/Relu:0' shape=(1, 256, 256, 64) dtype=float32>),.....]
        """
        _, endpoints_dict = network_fn(tf.concat([images, images_1], 0))
        features = []
        features_1 = []
        """ 依次计算 style layer 的 gram matrix 值"""
        for layer in FLAGS.style_layers:
            feature, feature_1 = tf.split(endpoints_dict[layer], 2, 0)
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            feature_1 = tf.squeeze(gram(feature_1), [0])  # remove the batch dimension
            features.append(feature)
            features_1.append(feature_1)

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
            save_file_1 = 'generated/target_style_' + FLAGS.naming_1 + '.jpg'

            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            with open(save_file_1, 'wb') as f:
                target_image_1 = image_unprocessing_fn(images_1[0, :])
                value_1 = tf.image.encode_jpeg(tf.cast(target_image_1, tf.uint8))
                f.write(sess.run(value_1))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file_1)
            return sess.run([features, features_1])

def style_loss_single(endpoints_dict, style_layers, style_strength, style_features_t, style_weight):
    # gram_size = 64 * 64 / 128 * 128 / 256 * 256 / 512 * 512
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_content_list = tf.split(endpoints_dict[layer], 16, 0)
        size = tf.size(generated_content_list[0])
        layer_style_loss = 0
        batch = len(style_strength)
        for i in range(batch):
            layer_style_loss += tf.nn.l2_loss(gram(generated_content_list[i]) - (style_strength[i] * style_gram + (1 - style_strength[i]) * gram(generated_content_list[i + batch]))) * 2 / tf.to_float(size)
        layer_style_loss = layer_style_loss / batch

        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss * style_weight, style_loss_summary

def style_loss_mixed(endpoints_dict, style_layers, style_strength, style_features_t, style_features_t_1, style_weight, style_weight_1):
    # gram_size = 64 * 64 / 128 * 128 / 256 * 256 / 512 * 512
    style_loss = 0
    style_loss_summary = {}
    style_weight_sqrt = tf.sqrt(style_weight)
    style_weight_sqrt_1 = tf.sqrt(style_weight_1)

    for style_gram, style_gram_1, layer in zip(style_features_t, style_features_t_1, style_layers):
        generated_content_list = tf.split(endpoints_dict[layer], 16, 0)
        size = tf.size(generated_content_list[0])
        layer_style_loss = 0
        batch = len(style_strength)
        for i in range(batch):
            layer_style_loss += tf.nn.l2_loss(
                    (style_weight_sqrt + style_weight_sqrt_1)/2 * gram(generated_content_list[i]) - 
                    (style_weight_sqrt* style_strength[i] * style_gram + style_weight_sqrt_1 * (1 - style_strength[i]) * style_gram_1)
                    ) * 2 / tf.to_float(size)
        layer_style_loss = layer_style_loss / batch

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
init
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

    """ variables_to_restore 只保留卷积层，不保留全连接层 """
    variables_to_restore = []
    for var in slim.get_model_variables():
        variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)

def mean_image_subtraction(image, means):
    image = tf.to_float(image)

    num_channels = 3
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)
