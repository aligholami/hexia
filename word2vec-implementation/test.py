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


# Load and preprocess the data
text_corpus = open('../ptb.train.txt', 'r').read()
text_corpus = text_corpus.lower()
num_corpus_words = get_num_words(text_corpus)
word_vocab = get_word_vocabulary(text_corpus)
word_vocab_size = len(word_vocab)
batch_size = 16
window_size = 3

num_steps = int(num_corpus_words/batch_size)

for step in range(5):
    print("Result for step: {0}".format(step))
    print(generate_encoder_mini_batch(batch_size, step, window_size))
