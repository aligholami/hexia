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

    word_one_hot_vec = np.zeros(shape=(len(vocab)))
    for ix, _word in enumerate(vocab):
        if _word == word:
            word_one_hot_vec[ix] = 1

    return word_one_hot_vec

def corpus_to_vec(text_corpus, time_steps):

    word_vocab = get_word_vocabulary(text_corpus)
    word_vocab_size = len(word_vocab)

    wordlist = text_corpus.split(' ')
    num_vectors = int(len(wordlist) / time_steps)

    # Our word vocab size will be similar to our one hot vectors in terms of dims
    vectorized_corpus = np.zeros(shape=(num_vectors, word_vocab_size), dtype=float)
    vectorized_labels = np.zeros(shape=(num_vectors, word_vocab_size), dtype=float)

    for vec_idx in range(num_vectors):
        try:
            selected_word = wordlist[vec_idx]
            selected_word_label = wordlist[vec_idx + 1]
            vectorized_corpus[vec_idx:] = get_vector(selected_word, word_vocab)
            vectorized_labels[vec_idx:] = get_vector(selected_word_label, word_vocab)
        except IndexError:
            pass


    return (vectorized_corpus, vectorized_labels)


# Load and preprocess the data
text_corpus = open('ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)

# No dense embeddings -> one-hot vectors only
# Hyperparams
num_epochs = 500
batch_size = 20
time_steps = 1
num_features = word_vocab_size
lstm_size = 150
learning_rate = 0.01

# Get array for each word in the corpus and place them in 10 timesteps formats
x_train, y_train = corpus_to_vec(text_corpus, time_steps)

print("Number of Words: ", num_corpus_words)
print("Number of Unique words:", word_vocab_size)

#############################
# Beginning of the TF Graph #
#############################

# Both data and labels are placeholders for scalability
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_features], name='x_input')
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_features], name='label_input')

# Define the model
lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, dtype=tf.float32)

# Initial LSTM memory
z_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

lstm_output, state = lstm(inputs=x, state=z_state)

# Final dense layer
logits = tf.layers.dense(inputs=lstm_output, units=num_features, activation=None, use_bias=True)

# Loss definition
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Grab the number of correct predictions for calculating the accuracy
correct_preds_vec = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
correct_preds = tf.reduce_sum(tf.cast(correct_preds_vec, tf.float32))

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    total_loss = 0
    num_vectors = x_train.shape[0]
    num_steps = int(num_vectors / batch_size)

    for epoch in range(num_epochs):

        x_train, y_train = corpus_to_vec(text_corpus, time_steps)
        correct_preds_in_epoch = 0
        loss_in_epoch = 0

        for step in range(num_steps):

            beg_idx = batch_size * step
            end_idx = beg_idx + batch_size
            
            # Run on each batch of data
            correct_predictions, preds, loss_val, _ = sess.run([correct_preds, logits, loss, opt], feed_dict={x:x_train[beg_idx:end_idx, :], y:y_train[beg_idx:end_idx, :]})
            loss_in_epoch += loss_val
            correct_preds_in_epoch += correct_predictions

        total_loss += loss_in_epoch
        accuracy = correct_preds_in_epoch / num_corpus_words * 100
        print("Loss at epoch {0}: {1}".format(epoch, loss_in_epoch))
        print("Accuracy at epoch {0}: {1}".format(epoch, accuracy))

    
    print("Total loss: {0}".format(total_loss))
        

