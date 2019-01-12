import tensorflow as tf
from tensorflow.contrib import slim

class FeatureExtractor:

    CONV1_NUM_FILTERS = 15
    CONV1_FILTER_SIZE = 3

    CONV2_NUM_FILTERS = 30
    CONV2_FILTER_SIZE = 3

    CONV3_NUM_FILTERS = 50
    CONV3_FILTER_SIZE = 3

    def __init__(self, keep_prob):

        self.keep_prob = keep_prob

    def generate_image_feature_map(self, cnn_input, name='Image_Feature_Extractor'):
        """
        Computation graph definition of the Convolutional Neural Network to extract image feature maps.
        """

        with tf.name_scope(name=name):

            conv1 = self.conv_bn_sc_relu(
                inputs=cnn_input,
                filters=self.CONV1_NUM_FILTERS,
                k_size=self.CONV1_FILTER_SIZE,
                stride=1,
                padding='SAME',
                scope_name='conv_1',
                keep_prob=self.keep_prob
            )

            conv2 = self.conv_bn_sc_relu(
                inputs=conv1,
                filters=self.CONV2_NUM_FILTERS,
                k_size=self.CONV2_FILTER_SIZE,
                stride=1,
                padding='SAME',
                scope_name='conv_2',
                keep_prob=self.keep_prob
            )

            conv3 = self.conv_bn_sc_relu(
                inputs=conv2,
                filters=self.CONV3_NUM_FILTERS,
                k_size=self.CONV3_FILTER_SIZE,
                stride=1,
                padding='SAME',
                scope_name='conv_3',
                keep_prob=self.keep_prob
            )

            flattened = tf.layers.flatten(
                inputs=conv3,
                name='flatten_image'
            )

            return flattened

    def conv_bn_sc_relu(self, inputs, filters, k_size, stride, padding, scope_name, keep_prob):
        """
        Building blocks for constructing a Convolutional Neural Network.
        """

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

            in_channels = inputs.shape[-1]

            kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters], initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

            conv = tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding, use_cudnn_on_gpu=True)

            # Perform a batch normalization
            norm = tf.layers.batch_normalization(inputs=conv, name='batch_norm')

            # Scale the normalized batch
            scaled_batch = self.scale(inputs=norm, scope_name='scale')

            # Perform a dropout on the input
            do_scaled_batch = tf.nn.dropout(
                x=scaled_batch,
                keep_prob=keep_prob,
                name='dropout'
            )

        # Perform a relu and return
        # return tf.nn.relu(scaled_batch + biases, name=scope.name)
        return tf.nn.relu(do_scaled_batch + biases, name='relu')

    def scale(self, inputs, scope_name):
        """
        Scale input using a simple linear transformation to learn variations better.
        """
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

            in_dim = inputs.shape[-1]
            alpha = tf.get_variable(name='alpha', shape=(in_dim, ), trainable=True)
            beta = tf.get_variable(name='beta', shape=(in_dim, ), trainable=True)

            scaled_input = alpha * inputs + beta

        return scaled_input

    def maxpool(self, inputs, k_size, stride, padding, scope_name):
        """
        Perform a max pooling operation for dimensionality reduction purposes on the feature maps.
        """
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            pool = tf.nn.max_pool(value=inputs, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding)

        return pool
