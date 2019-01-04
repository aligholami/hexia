<<<<<<< HEAD
import tensorflow as tf
from utils import load_embedding_from_disks

class WordVectorizer:

    def __init__(self, batch_size, glove_file_path, vocab_file_path):

        self.batch_size = batch_size
        self.vocab_file_path = vocab_file_path

        # Load embeddings to ram
        self.word_to_index, self.index_to_embedding = load_embedding_from_disks(glove_file_path, with_indexes=True)

    def load_trained_model_tensors(self, name='load_trained_glove'):

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

        with tf.name_scope(name=name):

            # TODO: Get embedding of words each
            # sentence_embedding = tf.reduce_mean(tf.nn.embedding_lookup(params=self.tf_embedding, ids=target_sentence))
            sentence_embedding = tf.reduce_mean(tf.nn.embedding_lookup(params=self.tf_embedding, ids=target_sentence), axis=1)

        return sentence_embedding
=======
import tensorflow as tf
from utils import load_embedding_from_disks

class WordVectorizer:

    def __init__(self, batch_size, glove_file_path):

        self.batch_size = batch_size

        # Load embeddings to ram
        self.word_to_index, self.index_to_embedding = load_embedding_from_disks(glove_file_path, with_indexes=True)

    def load_trained_model_tensors(self, name='load_trained_glove'):

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
        
        with tf.name_scope(name=name):

            target_sentence = tf.cast(target_sentence, tf.string)
            
            # Take out the words
            target_words = tf.string_split([target_sentence], delimiter=" ").values
            
            # TODO: Get embedding of words each            
            sentence_embedding = tf.reduce_mean(tf.nn.embedding_lookup(params=self.tf_embedding, ids=self.word_to_index[target_words[0]]))

        return sentence_embedding
>>>>>>> ad30bb6cf245299f788a8b049bc1857d82952da5
