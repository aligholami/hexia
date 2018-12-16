import tensorflow as tf

class FeatureExtractor:

    def __init__(self, keep_prob, flatten=True):
        
        self.keep_prob = keep_prob
        self.flatten = flatten

    def assemble_cnn_model(self, cnn_input):

        conv1 = self.conv_bn_sc_relu(
            inputs=cnn_input,
            filters=CONV1_NUM_FILTERS,
            k_size=CONV1_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_1',
            keep_prob=self.keep_prob
        )
    
        conv2 = self.conv_bn_sc_relu(
            inputs=conv1,
            filters=CONV2_NUM_FILTERS,
            k_size=CONV2_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_2',
            keep_prob=self.keep_prob
        )

        conv3 = self.conv_bn_sc_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_3',
            keep_prob=self.keep_prob
        )

        if(self.flatten == False):
            return conv3

        else:
            flattened = tf.layers.flatten(
                inputs=conv3,
                name='flatten_input'
            )

            return flattened
    
    def conv_bn_sc_relu(self, inputs, filters, k_size, stride, padding, scope_name, keep_prob):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

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

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

            in_dim = inputs.shape[-1]
            alpha = tf.get_variable(name='alpha', shape=(in_dim, ), trainable=True)
            beta = tf.get_variable(name='beta', shape=(in_dim, ), trainable=True)

            scaled_input = alpha * inputs + beta

        return scaled_input

    def maxpool(self, inputs, k_size, stride, padding, scope_name):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(value=inputs, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding)
            
        return pool

