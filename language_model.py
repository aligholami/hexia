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

# Load and preprocess the data
text_corpus = open('ptb.train.txt', 'r').read()
train_data_num_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)

# No dense embeddings -> one-hot vectors only
# Hyperparams
batch_size = 4
time_steps = len(vocab)
num_features = len(vocab)
lstm_size = 128

#############################
# Beginning of the TF Graph #
#############################
x = tf.placeholder(dtype=tf.float32, shape=[time_steps, batch_size, num_features], name='RNN_Input')

# Define the model
lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, dtype=tf.float32)

# Initial LSTM memory
z_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

output, state = lstm(inputs=word_batch, state=z_state)

# Final dense layer
logits = tf.layers.dense(inputs=output, units=num_features, activation=None, use_bias=True)

# Define loss
loss = tf.nn.softmax_cross_entropy_with_logits(logits, )




