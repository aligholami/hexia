import tensorflow as tf

def get_num_words(sentence):

    num_words = 0

    for idx, ch in enumerate(sentence):
        if ch == ' ':
            num_words += 1
        else:
            pass
    return num_words

def get_word_vocabulary(sentence):
    

BATCH_SIZE = 4
NUM_FEATURES = 5
TIME_STEPS = 5

train_data = open('ptb.train.txt', 'r').read()
train_data_num_words = get_num_words(train_data)

x = tf.placeholder(dtype=tf.float32, shape=[TIME_STEPS, BATCH_SIZE, NUM_FEATURES], name='RNN_Input')

print(train_data_num_words)