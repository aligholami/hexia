import tensorflow as tf

class FeatureExtractor:

    def __init__():
        pass;
        
    def assemble_cnn_model(self, cnn_input):
        
    
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



