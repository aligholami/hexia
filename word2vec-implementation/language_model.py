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
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)

# No dense embeddings -> one-hot vectors only
# Hyperparams
num_epochs = 500
batch_size = 200
time_steps = 1
vocabulary_size = word_vocab_size
embedding_size = 300
lstm_size = 50
learning_rate = 0.1

# Get array for each word in the corpus and place them in 10 timesteps formats
x_train, y_train = corpus_to_vec(text_corpus, time_steps)

print("Number of Words: ", num_corpus_words)
print("Number of Unique words:", word_vocab_size)

#############################
# Beginning of the TF Graph #
#############################

# The global step
g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# Both data and labels are placeholders for scalability
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, embedding_size], name='x_input')
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, embedding_size], name='label_input')

# Define the model
lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, dtype=tf.float32)

# Initial LSTM memory
z_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

lstm_output, state = lstm(inputs=x, state=z_state)

# Final dense layer
logits = tf.layers.dense(inputs=lstm_output, units=embedding_size, activation=None, use_bias=True)

# Loss definition & Visualization
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar('cross_entropy', cross_entropy)

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=g_step)

# Grab the number of correct predictions for calculating the accuracy
shape_t = tf.shape(logits)

# Compute accuracy
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))
tf.summary.scalar('accuracy', acc_op)

merged_summary = tf.summary.merge_all()

with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Initialize the file writer for Tensorboard
    visualizer = tf.summary.FileWriter('../visualization/word2vec')
    visualizer.add_graph(sess.graph)

    total_loss = 0
    num_vectors = x_train.shape[0]
    num_steps = int(num_vectors / batch_size)

    for epoch in range(num_epochs):

        x_train, y_train = corpus_to_vec(text_corpus, time_steps)
        epoch_accuracy = 0
        loss_in_epoch = 0

        for step in range(num_steps):

            beg_idx = batch_size * step
            end_idx = beg_idx + batch_size
            
            # Run on each batch of data
            accuracy, preds, loss_val, _, visualizer_summary = sess.run([acc_op, logits, cross_entropy, opt, merged_summary], feed_dict={x:x_train[beg_idx:end_idx, :], y:y_train[beg_idx:end_idx, :]})
            loss_in_epoch += loss_val
            epoch_accuracy += accuracy

        total_loss += loss_in_epoch
        epoch_accuracy = epoch_accuracy / num_steps
        print("Loss at epoch {0}: {1}".format(epoch, loss_in_epoch))
        print("Accuracy at epoch {0}: {1}".format(epoch, epoch_accuracy))

        # Write to visualization
        visualizer.add_summary(visualizer_summary, global_step=g_step.eval())

    
    print("Total loss: {0}".format(total_loss))
        

