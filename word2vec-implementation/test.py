import tensorflow as tf
import numpy as np
import math

def get_ix_to_word(vocab):

    ix_to_word = {ix:word  for ix, word in enumerate(vocab)}

    return ix_to_word

def get_word_to_ix(vocab):

    word_to_ix = {word:ix for ix, word in enumerate(vocab)} 

    return word_to_ix

def integerize_corpus(corpus):
    
    word_vocab_ = get_word_vocabulary(corpus)
    corpus_char_list = corpus.split(' ')

    
    integerize_corpus = []
    for char in corpus_char_list:
        char_index = 
        integerize_corpus.append(
    integerized_co
    for char in corpus_chars:


    integerized_corpus = 

def generate_encoder_mini_batch(batch_size, step, window_size):

    global text_corpus
    global num_corpus_words

    batch_train = np.zeros(shape=[batch_size], dtype=np.int32)
    batch_labels = np.zeros(shape=[batch_size, 1], dtype=np.int32)

    # Convert text corpus to integer
    #################################
    corpus_integerized = integerize_corpus(text_corpus)

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


# Load and preprocess the data
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)

print(generate_encoder_mini_batch(5, 1, 3))
