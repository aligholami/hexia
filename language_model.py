import tensorflow as tf
import numpy as np

np.set_printoptions(threshold=np.nan)

def get_num_words(sequence):

    num_words = 0

    for idx, ch in enumerate(sequence):
        if ch == ' ':
            num_words += 1
        else:
            pass
    return num_words

def get_word_vocabulary(sequence):

    word_vocab = {}
    wordlist = sequence.split(' ')

    for word in wordlist:
        if word not in word_vocab:
            word_vocab[word] = 0
        else:
            word_vocab[word] += 1
    
    return word_vocab

def get_ix_to_word(vocab):

    ix_to_word = {ix:word  for ix, word in enumerate(vocab)}

    return ix_to_word

def get_word_to_ix(vocab):

    word_to_ix = {word:ix for ix, word in enumerate(vocab)} 

    return word_to_ix


def get_vector(word, vocab):

    word_one_hot_vec = np.zeros(shape=(len(vocab), 1))
    for ix, _word in enumerate(vocab):
        if _word == word:
            word_one_hot_vec[ix] = 1

    return word_one_hot_vec

BATCH_SIZE = 4
NUM_FEATURES = 5
TIME_STEPS = 5

train_data = open('ptb.train.txt', 'r').read()
train_data_num_words = get_num_words(train_data)

word_vocab = get_word_vocabulary(train_data)
one_hot_word = get_vector('they', word_vocab)

x = tf.placeholder(dtype=tf.float32, shape=[TIME_STEPS, BATCH_SIZE, NUM_FEATURES], name='RNN_Input')

