import tensorflow as tf
import numpy as np
from feature_extractor import FeatureExtractor
from word_vectorizer import WordVectorizer
from classifier import Classifier
from data_generator import DataGenerator
from utils import load_embedding_from_disks

class VQA_SAN:

    PATH_TO_TRAIN_IMAGES = '../data/train/images/full-image-dir'
    PATH_TO_TRAIN_QUESTIONS = '../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    PATH_TO_TRAIN_ANSWERS = '../data/train/answers/v2_mscoco_train2014_annotations.json'
    PATH_TO_TRAINED_GLOVE = '../models/GloVe/glove.6B.50d.txt'
    PATH_TO_WORD_VOCAB = '../models/GloVe/vocab_only.txt'
    PATH_TO_VISUALIZATION_GRAPHS = '../visualization/'

    BATCH_SIZE = 1
    NUM_CLASSES = 3     # Yes / Maybe / No
    LEARNING_RATE = 0.0001

    def __init__(self):

        # Define global step variable
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.skip_steps = 100

    def get_data(self):

        # Setup the generator
        train_generator = DataGenerator(image_path=self.PATH_TO_TRAIN_IMAGES,
                        q_path=self.PATH_TO_TRAIN_QUESTIONS,
                        a_path=self.PATH_TO_TRAIN_ANSWERS,
                        image_rescale=1, image_horizontal_flip=False, image_target_size=(150, 150))

        train_data_generator = lambda: train_generator.mini_batch_generator()

        train_data = tf.data.Dataset.from_generator(
            generator=train_data_generator,
            output_types=(tf.float32, tf.string, tf.string, tf.float32),
            output_shapes=(tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)),
        ).batch(self.BATCH_SIZE)

        # Load words vocab table for quick lookup
        word_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.PATH_TO_WORD_VOCAB, num_oov_buckets=1)

        iterator = train_data.make_initializable_iterator()

        img, question, answer, self.label = iterator.get_next()
        
        # Merge question and answer
        sentence = tf.strings.join([question, answer], separator=' ')

        with tf.name_scope('sentence_splitter'):
            # question = tf.map_fn(lambda x: tf.string_split([x], delimiter=' ').values, question, dtype=tf.string)
            # answer = tf.map_fn(lambda x: tf.string_split([x], delimiter=' ').values, answer, dtype=tf.string)
            sentence = tf.map_fn(lambda x: tf.string_split([x], delimiter=' ').values, sentence, dtype=tf.string)
        
        with tf.name_scope('word_table_lookup'):
            # self.question = tf.map_fn(lambda x: tf.cast(word_table.lookup(x), tf.int32), question, dtype=tf.int32)
            # self.answer = tf.map_fn(lambda x: tf.cast(word_table.lookup(x), tf.int32), answer, dtype=tf.int32)
            self.sentence = tf.map_fn(lambda x: tf.cast(word_table.lookup(x), tf.int32), sentence, dtype=tf.int32)

        # Preapre image for a CNN pass
        self.img = tf.reshape(img, [-1, 64, 64, 3])

        # Add iterators to the graph
        self.train_init = iterator.make_initializer(train_data)           # Train iterator
        self.validation_init = iterator.make_initializer(train_data)      # Validation iterator


    def loss(self):

        with tf.name_scope('loss'):

            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=(self.predictions + 1e-8))

            # Loss is mean of error on all dimensions
            self.loss_val = tf.reduce_mean(cross_entropy_loss, name='loss')

    def optimize(self):

        with tf.name_scope('SGDOptimizer'):

            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.LEARNING_RATE,
                name='SGDOptimizer'
            ).minimize(self.loss_val, global_step=self.g_step)

    def eval(self):

        with tf.name_scope('Accuracy'):

            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.label, 1)), tf.float32))

    def summary(self):

        with tf.name_scope('graph_summary'):

            tf.summary.scalar('num_t_preds', self.accuracy)
            # tf.summary.histogram('loss_histogram', self.loss_val)
            tf.summary.scalar('loss_value', self.loss_val)

            self.summary = tf.summary.merge_all()


    def build_model(self):

        # Setup input data pipeline
        self.get_data()

        # Feature extraction for the image
        feature_extractor = FeatureExtractor(keep_prob=1.0)

        # Word embeddings
        word_vectorizer = WordVectorizer(batch_size=self.BATCH_SIZE,
                                         glove_file_path=self.PATH_TO_TRAINED_GLOVE,
                                         vocab_file_path=self.PATH_TO_WORD_VOCAB)


        self.embedding_init = word_vectorizer.load_trained_model_tensors()

        # Classifer
        classifier = Classifier(self.NUM_CLASSES)

        # Obtain image feature maps
        self.image_feature_map = feature_extractor.generate_image_feature_map(self.img)

        # # Obtain answer embeddings
        # self.answer_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.answer))

        # # Obtain sentence embeddings
        # self.question_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.question))

        # Obtain sentence embeddings
        self.sentence_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.sentence))

        # Concatenate image feature map and sentence feature maps
        # self.iqa_vector = tf.concat(values=[self.image_feature_map, self.question_glove_vector, self.answer_glove_vector], axis=1)

        # self.iqa_vector = tf.concat(values=[self.question_glove_vector, self.image_feature_map], axis=1)

        # self.iqa_vector = tf.concat(values=[iqa_vector, self.answer_glove_vector], axis=1)

        # Classify the concatenated vector
        self.predictions = classifier.classify_input(self.sentence_glove_vector)

        # Setup loss function
        self.loss()

        # Setup optimizer
        self.optimize()

        # Model accuracy and evaluation
        self.eval()

        # Training/Testing summary
        self.summary()


    def train_one_epoch(self, init, sess, writer, step):

        # Initialize input data based on the training or validation
        sess.run(init)
        total_loss = 0

        try:
            while True:
                # Get accuracy, loss value and optimize the network + summary of validation
                step_accuracy, step_loss, _, step_summary= sess.run([self.accuracy, self.loss_val, self.opt, self.summary])

                step += 1
                total_loss += step_loss

                if((step + 1) % self.skip_steps == 0):
                    print('Loss at step {}: {}'.format(step, step_loss))
                    writer.add_summary(step_summary, global_step=step)

        except tf.errors.OutOfRangeError:
            pass;

    def validate(self, init, sess, writer, step, epoch):

        # Initialize input data based on the training or validation
        self.init = init

        total_loss = 0

        try:
            while True:
                # Get accuracy and summary of validation
                step_accuracy, step_loss, step_summary = sess.run([self.accuracy, self.loss_val, self.summary])

                step += 1
                total_loss += step_loss

        except tf.errors.OutOfRangeError:
            pass;

    def train_and_validate(self, batch_size, num_epochs):

        # Tensorflow writer for graphs and summary saving
        train_writer = tf.summary.FileWriter(self.PATH_TO_VISUALIZATION_GRAPHS, tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.PATH_TO_VISUALIZATION_GRAPHS, tf.get_default_graph())


        with tf.Session().as_default() as sess:

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Initialize tables
            sess.run(tf.tables_initializer())

            # Load GloVe embeddings
            sess.run(self.embedding_init)

            step = self.g_step.eval()

            # Train multiple epochs
            for epoch in range(num_epochs):

                step = self.train_one_epoch(
                    init=self.train_init,
                    sess=sess,
                    writer=train_writer,
                    step=step
                )

                # Validate the learned model
                # self.validate(
                #     init=self.validation_init,
                #     sess=sess,
                #     writer=validation_writer,
                #     step=step
                # )
