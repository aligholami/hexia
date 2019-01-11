import tensorflow as tf


class Classifier:

    def __init__(self, num_classes):

        self.num_classes = num_classes

    def classify_input(self, feature_vector):
        """
        Computation graph definition of a simple multi layer perceptron.
        """

        with tf.name_scope("Classifier"):

            fc_1 = tf.layers.dense(
                inputs=feature_vector,
                units=120
            )

            fc_2 = tf.layers.dense(
                inputs=fc_1,
                units=64
            )

            predictions = tf.layers.dense(
                inputs=fc_2,
                units=self.num_classes,
            )

        return predictions
