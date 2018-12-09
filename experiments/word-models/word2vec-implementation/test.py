import tensorflow as tf
import numpy as np
import math

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

# Load and preprocess the data
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)
batch_size = 16
window_size = 3
embedding_epochs = 500

num_steps = int(num_corpus_words/batch_size)

# get the word embeddings
em_inputs, em_labels, embeddings, optimize, loss = word_embeddings(word_vocab_size)

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
