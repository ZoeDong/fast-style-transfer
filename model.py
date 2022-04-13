import tensorflow as tf
from tensorflow.contrib.slim import instance_norm

def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)

'''
def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
'''

def batch_norm(x, size, training, decay=0.999):
    training = tf.constant(training, tf.bool)
    # size = x.get_shape().as_list()[-1] # 获得最后一个通道数，即features的个数
    beta = tf.Variable(tf.zeros([size]), name='beta') # beta通常初始化为0
    scale = tf.Variable(tf.ones([size]), name='scale') # scale 通常初始化为1
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2]) # 求出前三个通道的均值和标准差，此时的维度 = (chanel,)
    # 第二个参数：想要 normalize 的维度, [0] 代表 batch 维度
    # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) # pop_mean表示上一次的均值，batch_mean表示当前的x的均值 
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay)) # pop_var表示上一次的标准差， batch_var表示当前的x的标准差

    def batch_statistics(): # training 更新population_mean & population_var
        with tf.control_dependencies([train_mean, train_var]): # 确保获取更新后的train_mean,train_var
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')

    def population_statistics(): # inference 使用population_mean & population_var
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics) # tf.cond：流程控制，training==True，则执行参数2，否则执行参数3


def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero


def residual(x, filters, kernel, strides, style_strength, training):
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides) # shape=(4, 69, 69, 128)

        layer_strength = tf.Variable(tf.constant(1.0), trainable=True) # 添加一个可训练参数

        if training:
            style_strength = [0, 0, 1/3, 1/3, 2/3, 2/3, 1, 1]
            # style_strength = [0, 0, 0, 0, 0, 0, 0, 0]
            batch_size = tf.shape(conv2)[0].eval()
            residual = []
            cnt = 0
            for x_each, conv2_each in zip(tf.unstack(x, axis=0, num=batch_size), tf.unstack(conv2, axis=0, num=batch_size)):
                print(">>>>>>>>>>>>>>>>>>>>> style_strength[cnt] = ", style_strength[cnt])
                strength = style_strength[cnt] * layer_strength # 可训练参数和style strength绑定
                strength = 2 * tf.abs(strength) / (1 + tf.abs(strength)) # 限制范围在[0,2)
                residual.append(x_each + strength * conv2_each) # zoe S5 v2: layer_strength shape = [1,] 改回标量
                cnt += 1
            residual = tf.stack(residual)
        else:
            strength = style_strength * layer_strength # 可训练参数和style strength绑定
            strength = 2 * tf.abs(strength) / (1 + tf.abs(strength)) # 限制范围在[0,2)
            residual = x + strength * conv2
        return residual


def net(image, style_strength, training):
    # Less border effects when padding a little before passing through ..
    # print("*********** before image.shape:",image.shape)/
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    # print("*********** after  image.shape:",image.shape)

    is_instance_norm = True
    if is_instance_norm:
        # image: Tensor("MirrorPad:0", shape=(4, 276, 276, 3), dtype=float32)
        # conv1: Tensor("conv1/Select:0", shape=(4, 276, 276, 32), dtype=float32)
        # conv2: Tensor("conv2/Select:0", shape=(4, 138, 138, 64), dtype=float32)
        # conv3: Tensor("conv3/Select:0", shape=(4, 69, 69, 128), dtype=float32)
        # res1: Tensor("res1/residual/add_1:0", shape=(4, 69, 69, 128), dtype=float32)
        # res2: Tensor("res2/residual/add_1:0", shape=(4, 69, 69, 128), dtype=float32)
        # res3: Tensor("res3/residual/add_1:0", shape=(4, 69, 69, 128), dtype=float32)
        # res4: Tensor("res4/residual/add_1:0", shape=(4, 69, 69, 128), dtype=float32)
        # res5: Tensor("res5/residual/add_1:0", shape=(4, 69, 69, 128), dtype=float32)
        # deconv1: Tensor("deconv1/Select:0", shape=(4, 138, 138, 64), dtype=float32)
        # deconv2: Tensor("deconv2/Select:0", shape=(4, 276, 276, 32), dtype=float32)
        # deconv3: Tensor("deconv3/Tanh:0", shape=(4, 276, 276, 3), dtype=float32)
        # y: Tensor("Slice_1:0", shape=(4, 256, 256, 3), dtype=float32)

        with tf.variable_scope('conv1'):
            conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
        with tf.variable_scope('conv2'):
            conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
        with tf.variable_scope('conv3'):
            conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
        with tf.variable_scope('res1'):
            res1 = residual(conv3, 128, 3, 1, style_strength, training)
        with tf.variable_scope('res2'):
            res2 = residual(res1, 128, 3, 1, style_strength, training)
        with tf.variable_scope('res3'):
            res3 = residual(res2, 128, 3, 1, style_strength, training)
        with tf.variable_scope('res4'):
            res4 = residual(res3, 128, 3, 1, style_strength, training)
        with tf.variable_scope('res5'):
            res5 = residual(res4, 128, 3, 1, style_strength, training)
        # print(res5.get_shape())
        with tf.variable_scope('deconv1'):
            # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
            deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
        with tf.variable_scope('deconv2'):
            # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
            deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
        with tf.variable_scope('deconv3'):
            # deconv_test = relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1)))
            deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1))) # Tensor("deconv3/Tanh:0", shape=(1, 476, 712, 3), dtype=float32)

    else:
        with tf.variable_scope('conv1'):
            # conv2d return: Tensor("conv1/conv/conv:0", shape=(4, 276, 276, 32), dtype=float32)
            conv1 = relu(batch_norm(conv2d(image, 3, 32, 9, 1), 32, training)) 
        with tf.variable_scope('conv2'):
            # conv2d return: Tensor("conv2/conv/conv:0", shape=(4, 138, 138, 64), dtype=float32)
            conv2 = relu(batch_norm(conv2d(conv1, 32, 64, 3, 2), 64, training))
        with tf.variable_scope('conv3'):
            # conv2d return: Tensor("conv3/conv/conv:0", shape=(4, 69, 69, 128), dtype=float32)
            conv3 = relu(batch_norm(conv2d(conv2, 64, 128, 3, 2), 128, training))
        with tf.variable_scope('res1'):
            res1 = residual(conv3, 128, 3, 1, style_strength)
        with tf.variable_scope('res2'):
            res2 = residual(res1, 128, 3, 1, style_strength)
        with tf.variable_scope('res3'):
            res3 = residual(res2, 128, 3, 1, style_strength)
        with tf.variable_scope('res4'):
            res4 = residual(res3, 128, 3, 1, style_strength)
        with tf.variable_scope('res5'):
            res5 = residual(res4, 128, 3, 1, style_strength)
        # print(res5.get_shape())
        with tf.variable_scope('deconv1'):
            # resize_conv2d return: Tensor("deconv1/conv_transpose/conv/conv:0", shape=(4, 138, 138, 64), dtype=float32)
            deconv1 = relu(batch_norm(resize_conv2d(res5, 128, 64, 3, 2, training), 64, training))
        with tf.variable_scope('deconv2'):
            # resize_conv2d return: Tensor("deconv2/conv_transpose/conv/conv:0", shape=(4, 276, 276, 32), dtype=float32)
            deconv2 = relu(batch_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training), 32, training))
        with tf.variable_scope('deconv3'):
            # conv2d return: Tensor("deconv3/conv/conv:0", shape=(4, 276, 276, 3), dtype=float32)
            deconv3 = tf.nn.tanh(batch_norm(conv2d(deconv2, 32, 3, 9, 1), 3, training))
    y = (deconv3 + 1) * 127.5
    # print("**************** y:",y)
    # y = deconv3

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y
