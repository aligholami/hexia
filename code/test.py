from word_vectorizer import WordVectorizer
import tensorflow as tf
vectizer = WordVectorizer(batch_size=2, glove_file_path='../models/GloVe/glove.6B.50d.txt', vocab_file_path='../models/GloVe/vocab_only.txt')

initializer = vectizer.load_trained_model_tensors()
embedding = vectizer.generate_sentence_vector('hello')

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Transfer embedding to GPU
    sess.run(initializer)

    # Get target embeddings
    print("Target embeddings: ", sess.run(embedding))
