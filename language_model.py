import tensorflow as tf
import numpy as np
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

def get_ix_to_char(vocab):

    ix_to_char = {ix:ch  for ix, ch in enumerate(vocab)}
    
    return ix_to_char

def get_char_to_ix(vocab):

    char_to_ix = {ch:ix for ix, ch in enumerate(vocab)} 

    return char_to_ix

def get_vector(word, vocab):

    word_one_hot_vec = np.zeros(shape=(len(vocab), 1))


BATCH_SIZE = 4
NUM_FEATURES = 5
TIME_STEPS = 5

train_data = open('ptb.train.txt', 'r').read()
train_data_num_words = get_num_words(train_data)

word_vocab = get_word_vocabulary(train_data)

for ix, ch in enumerate(word_vocab):
    print(ix, ":", ch)


x = tf.placeholder(dtype=tf.float32, shape=[TIME_STEPS, BATCH_SIZE, NUM_FEATURES], name='RNN_Input')

