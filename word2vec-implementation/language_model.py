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
    encoder_learning_rate = 0.001
    encoder_batch_size = 144

    # Integer representation of the inputs
    train_inputs = tf.placeholder(dtype=tf.int32, shape=[encoder_batch_size])
    train_labels = tf.placeholder(dtype=tf.int32, shape=[encoder_batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedded = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)

    # Noise contrastive loss weight vector and bias
    with tf.name_scope(name='weights'):
        nce_weights = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros(shape=[vocabulary_size]))
    
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

    return train_inputs, train_labels, embeddings, optimize, loss

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

    return x, y, acc_op, logits, cross_entropy, opt, merged_summary


def generate_encoder_mini_batch(batch_size, step, window_size):

    global text_corpus
    global num_corpus_words

    batch_train = []
    batch_labels = []

    #################################
    # Convert text corpus to integer
    words = get_word_vocabulary(text_corpus)

    dictionary = dict()

    for word in words:
        dictionary[word] = len(dictionary)

    data = []

    splitized_corpus = text_corpus.split()
    for word in splitized_corpus:
        index = dictionary.get(word, 0)
        data.append(index)
    
    # Converted data into int is ready
    ##################################

    local_step = step

    for sample in range(batch_size):

        # While !EOF
        if((local_step + window_size + 1) <= num_corpus_words):

            context_idx = [local_step + window_idx for window_idx in range(window_size)]
            target_idx = local_step + window_size + 1

            # Add context words
            # What did you do?
            # Considering you as target, and the rest as context
            # This will be stored as (what, you), (did, you), (do, you)
            for word_idx in context_idx:
                batch_train.append(data[word_idx])
                batch_labels.append(data[target_idx])

        local_step += 1
    
    # Convert to numpy array and return
    return np.asarray(batch_train, dtype=np.int32), np.reshape(np.asarray(batch_labels, dtype=np.int32), newshape=[len(batch_labels), 1])


def generate_lstm_mini_batch(word_embeddings, batch_size, step, _, window_size):

    global text_corpus
    global num_corpus_words
    global embedding_size

    batch_train = []
    batch_labels = []
    local_step = step

    words = get_word_vocabulary(text_corpus)

    dictionary = dict()

    for word in words:
        dictionary[word] = len(dictionary)

    splitized_corpus = text_corpus.split()
    for word in splitized_corpus:
        index = dictionary.get(word, 0)
        data.append(index)

    
    for sample in range(batch_size):

        # While !EOF
        if((local_step + window_size + 1) <= num_corpus_words):

            context_integerized = []
            target_integerized = 0

            # Get words integers
            context_idx = [local_step + window_idx for window_idx in range(window_size)]
            for word in context_idx:
                context_integerized.append(dictionary.get(word, 0))

            target_idx = local_step + window_size + 1
            target_integerized = dictionary.get(target_idx, 0)

            # Add context words
            # What did you do?
            # Considering you as target, and the rest as context
            # This will be stored as (what, you), (did, you), (do, you)
            for word_int in context_integerized:
                batch_train.append(word_embeddings[word_int:])
            
            batch_labels.append(word_embeddings[target_integerized:])

        local_step += 1
    
    return np.asarray(np.reshape(batch_train, newshape=[batch_size, embedding_size])), np.asarray(np.reshape(batch_labels, newshape=[batch_size, embedding_size]))

# Load and preprocess the data
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)
batch_size = 16
window_size = 3
embedding_epochs = 500
embedding_size = 300
learning_rate = 0.01

###############################
# Build the computation graph #
###############################

# get the word embeddings
em_inputs, em_labels, embeddings, optimize, loss = word_embeddings(vocabulary_size=word_vocab_size)

# train the classfier
lstm_inputs, lstm_labels, lstm_acc, lstm_outputs, lstm_loss, lstm_opt, lstm_summary = lstm_inference(embeddings)

num_steps = int(num_corpus_words/batch_size)

embedding_sess = None

# Train the word embedding model
with tf.Session() as embedding_sess:

    embedding_sess.run(tf.initialize_all_variables())

    embedding_num_steps = int(num_corpus_words / (batch_size * window_size))

    for epoch in range(embedding_epochs):
        epoch_loss = 0
        for step in range(embedding_num_steps):

            real_batch_size = batch_size * window_size
            x_train, y_train = generate_encoder_mini_batch(real_batch_size, step, window_size)
            x_train = x_train / np.sqrt((np.sum(x_train**2)))
            y_train = y_train / np.sqrt((np.sum(y_train**2)))
            _, loss_val = embedding_sess.run([optimize, loss], feed_dict={em_inputs: x_train, em_labels: y_train})  
            epoch_loss += loss_val

        print("Loss value at epoch {0}:{1}".format(epoch, epoch_loss))

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

            x_train, y_train = generate_lstm_mini_batch(embeddings.eval(), batch_size, step, window_size)

            # Run on each batch of data
            accuracy, preds, loss_val, _, visualizer_summary = lstm_sess.run([lstm_acc, lstm_outputs, lstm_loss, lstm_opt, lstm_summary], feed_dict={lstm_inputs:x_train, lstm_labels:y_train]})
            loss_in_epoch += loss_val
            epoch_accuracy += accuracy

        total_loss += loss_in_epoch
        epoch_accuracy = epoch_accuracy / num_steps
        print("Loss at epoch {0}: {1}".format(epoch, loss_in_epoch))
        print("Accuracy at epoch {0}: {1}".format(epoch, epoch_accuracy))

        # Write to visualization
        visualizer.add_summary(visualizer_summary, global_step=g_step.eval())

    print("Total loss: {0}".format(total_loss))
        

