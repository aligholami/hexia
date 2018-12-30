import tensorflow as tf
from utils import load_embedding_from_disks

class WordVectorizer:

    def __init__(self, batch_size, glove_file_path):

        self.batch_size = batch_size
        self.emebedding_matrix_shape = emebedding_matrix_shape
        self.word_to_index, self.index_to_embedding = load_embedding_from_disks(glove_file_path, with_indexes=True)

    def generate_word_vector(self, target_word, name='word_vector_generator'):
        
        with tf.name_scope(name=name):

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
                ids=target_word
            )

        return vectorized_representation

    def generate_question_vector(self, target_sentence, name='sentence_vector_generator'):
        
        with tf.name_scope(name=name):

            # define the variable that holds the embedding
            tf_embedding = tf.Variable(
                tf.constant(0.0, shape=self.index_to_embedding.shape),
                trainable=False,
                name='sentence_embedding'
            )

            target_words = tf.string_split(target_sentence, delimiter="")
            sentence_mean_embedding = tf.reduce_mean(tf.map_fn(lambda target_word: tf.nn.embedding_lookup(params=tf_embedding, ids=target_word), target_words))

        return sentence_mean_embedding
