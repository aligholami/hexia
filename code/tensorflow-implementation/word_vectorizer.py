import tensorflow as tf
from utils import load_embedding_from_disks

class WordVectorizer:

    def __init__(self, batch_size, glove_file_path, vocab_file_path):

        self.batch_size = batch_size
        self.vocab_file_path = vocab_file_path

        # Load embeddings to ram
        self.word_to_index, self.index_to_embedding = load_embedding_from_disks(glove_file_path, with_indexes=True)

    def load_trained_model_tensors(self, name='load_trained_glove'):
        """
        Load the pre-trained language model to GDDR memory for direct Tensorflow lookups.
        """

        with tf.name_scope(name=name):

            # define the variable that holds the embedding
            self.tf_embedding = tf.Variable(
                tf.constant(0.0, shape=self.index_to_embedding.shape),
                trainable=False,
                name='sentence_embedding'
            ) 

            # Load embeddings to gddr while running tf_embedding_init
            tf_embedding_on_gddr = tf.constant(value=self.index_to_embedding, dtype=tf.float32, shape=self.index_to_embedding.shape)
            tf_embedding_init = self.tf_embedding.assign(tf_embedding_on_gddr)

        return tf_embedding_init

    def generate_sentence_vector(self, target_sentence, name='sentence_vector_generator'):
        """
        Computation graph definition for a word lookup in the loaded language model in GDDR.
        """
        
        with tf.name_scope(name=name):

            # TODO: Get embedding of words each
            # sentence_embedding = tf.reduce_mean(tf.nn.embedding_lookup(params=self.tf_embedding, ids=target_sentence))
            sentence_embedding = tf.reduce_mean(tf.nn.embedding_lookup(params=self.tf_embedding, ids=target_sentence), axis=1)

        return sentence_embedding
