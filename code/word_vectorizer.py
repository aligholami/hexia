import tensorflow as tf
from utils import load_embedding_from_disks

class WordVectorizer:

    def __init__(self, batch_size, glove_file_path):

        self.batch_size = batch_size
        self.emebedding_matrix_shape = emebedding_matrix_shape
        self.word_to_index, self.index_to_embedding = load_embedding_from_disks(glove_file_path, with_indexes=True)

    def generate_word_vector(self, target_words):
        
        # define the variable that holds the embedding
        tf_embedding = tf.Variable(
            tf.constant(0.0, shape=self.index_to_embedding.shape),
            trainable=False,
            name='word_embedding'
        )

        # lookup and return the desired embeddings
        # Note: target_words is a placeholder
        vectorized_representation = tf.nn.embedding_lookup(
            params=tf_embedding,
            ids=target_words
        )

        return vectorized_representation

