import tensorflow as tf


class Classifier:

    def __init__(self, num_classes):

        self.num_classes = num_classes

    def classify_input(self, feature_vector):
        """
        Computation graph definition of a simple multi layer perceptron.
        """
        
        predictions = tf.layers.dense(
            inputs=feature_vector,
            units=self.num_classes,
            name='Classifier'
        )

        return predictions
