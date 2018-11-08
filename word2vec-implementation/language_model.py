import tensorflow as tf
import numpy as np
import math

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

def word_embeddings(vocabulary_size):
    
    embedding_size = 300
    num_sampled = 64
    encoder_learning_rate = 0.01
    encoder_batch_size = 32

    # Integer representation of the inputs
    train_inputs = tf.placeholder(dtype=tf.int32, shape=[encoder_batch_size])
    train_labels = tf.placeholder(dtype=tf.int32, shape=[encoder_batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], -1.0, 1.0))
    embedded = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)

    # Noise contrastive loss weight vector and bias
    with tf.name_scope(name='weights'):
        nce_weights = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zers(shape=[vocabulary_size]))
    
    with tf.name_scope(name='loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embedded,
            num_sampled=num_sampled,
            num_classes=vocabulary_size
        ))

    with tf.name_scope(name='optimizer'):
        optimize = tf.train.GradientDescentOptimizer(learning_rate=encoder_learning_rate).minimize(loss)

    return train_inputs, train_labels, embeddings, optimize


def lstm_inference(word_embeddings):

    num_epochs = 500
    batch_size = 200
    vocabulary_size = word_vocab_size
    lstm_size = 50
    lstm_learning_rate = 0.1

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

    return acc_op, logits, cross_entropy, opt, merged_summary


def generate_encoder_mini_batch(batch_size, step, window_size):

    global text_corpus
    global num_corpus_words

    batch_train = np.zeros(shape=[batch_size], dtype=np.int32)
    batch_labels = np.zeros(shape=[batch_size, 1], dtype=np.int32)

    # Convert text corpus to integer
    #################################
    words = get_word_vocabulary(text_corpus)
    dictionary = dict()

    for word in words:
        dictionary[word] = len(dictionary)

    data = list()

    for word in words:
        index = dictionary.get(word, 0)
        data.append(index)

    local_step = step

    # While !EOF
    for sample in range(batch_size):

        if((local_step + window_size + 1) <= num_corpus_words):

            context_idx = [local_step + window_idx for window_idx in range(window_size)]
            target_idx = local_step + window_size + 1

            # Add context words
            batch_train = np.append(batch_train, np.asarray(text_corpus[context_idx]))
            batch_labels = np.append(batch_labels, np.assarray(text_corpus[target_idx]))

        local_step += 1

    return batch_train, batch_labels




def generate_lstm_mini_batch(batch_size, step, _):

    global text_corpus

    return batch_train, batch_labels




# Load and preprocess the data
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)


###############################
# Build the computation graph #
###############################

# get the word embeddings
em_input, em_labels, embeddings, optimize = word_embeddings()

# train the classfier
lstm_acc, lstm_outputs, lstm_loss, lstm_opt, lstm_summary = lstm_inference(embeddings)

# Train the word embedding model
with tf.Session() as embedding_sess:

    embedding_sess.run(tf.local_variables_intializer())
    embedding_sess.run(tf.global_variables_initializer())

    for epoch in range(embedding_epochs):

        for step in range(embedding_num_steps):

            x_train, y_train = generate_encoder_mini_batch(batch_size, step)
            embedding_sess.run([optimize], feed_dict={em_inputs: x_train, em_labels: y_train})

# Train the LSTM model
with tf.Session() as lstm_sess:

    lstm_sess.run(tf.local_variables_initializer())
    lstm_sess.run(tf.global_variables_initializer())

    # Initialize the file writer for Tensorboard
    visualizer = tf.summary.FileWriter('../visualization/word2vec')
    visualizer.add_graph(sess.graph)

    total_loss = 0
    num_vectors = x_train.shape[0]
    num_steps = int(num_vectors / batch_size)

    for epoch in range(num_epochs):
    
        epoch_accuracy = 0
        loss_in_epoch = 0

        for step in range(num_steps):

            x_train, y_train = generate_lstm_mini_batch(batch_size, step)
            
            # Run on each batch of data
            accuracy, preds, loss_val, _, visualizer_summary = lstm_sess.run([lstm_acc, lstm_outputs, lstm_loss, lstm_opt, lstm_summary], feed_dict={x:x_train, y:y_train]})
            loss_in_epoch += loss_val
            epoch_accuracy += accuracy

        total_loss += loss_in_epoch
        epoch_accuracy = epoch_accuracy / num_steps
        print("Loss at epoch {0}: {1}".format(epoch, loss_in_epoch))
        print("Accuracy at epoch {0}: {1}".format(epoch, epoch_accuracy))

        # Write to visualization
        visualizer.add_summary(visualizer_summary, global_step=g_step.eval())

    print("Total loss: {0}".format(total_loss))
        

