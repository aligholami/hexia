import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from tensorflow.python.client import timeline
from feature_extractor import FeatureExtractor
from word_vectorizer import WordVectorizer
from classifier import Classifier
from data_generator import DataGenerator

class VQA_SAN:

    PATH_TO_TRAIN_IMAGES = '../../data/train/images/'
    PATH_TO_TRAIN_QUESTIONS = '../../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    PATH_TO_TRAIN_ANSWERS = '../../data/train/answers/v2_mscoco_train2014_annotations.json'
    PATH_TO_VALIDATION_IMAGES = '../../data/validation/images/'
    PATH_TO_VALIDATION_QUESTIONS = '../../data/validation/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    PATH_TO_VALIDATION_ANSWERS = '../../data/validation/answers/v2_mscoco_val2014_annotations.json'
    PATH_TO_TRAINED_GLOVE = '../../models/GloVe/glove.6B.50d.txt'
    PATH_TO_WORD_VOCAB = '../../models/GloVe/vocab_only.txt'
    PATH_TO_TRAIN_VISUALIZATION_GRAPHS = '../../visualization/train'
    PATH_TO_VALIDATION_VISUALIZATION_GRAPHS = '../../visualization/validation'
    PATH_TO_MODEL_CHECKPOINTS = '../../models/checkpoints/baseline'
    PATH_TO_PRETRAINED_CNN_WEIGHTS = '../../models/ResNet'
    PATH_TO_TRAIN_PICKLE_FILES = '../../models/PickleFiles/train_data_items.txt'
    PATH_TO_VALIDATION_PICKLE_FILES = '../../models/PickleFiles/validation_data_items.txt'
    PATH_TO_TRACE_FILE = '../../visualization/trace/trace_f.ctf.json'

    TRAIN_INIT_CODE = 2
    VAL_INIT_CODE = 3

    BATCH_SIZE = 128
    NUM_CLASSES = 3     # Yes / Maybe / No
    LEARNING_RATE = 0.0001
    PREFETCH = 1
    NUM_PARALLEL_CALLS = 8
    IMAGE_SIZE = 64
    CLASS_LIST = ['Yes', 'Maybe', 'No']
    
    def __init__(self):

        # Define global step variable
        self.g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Define is_traning for variations in batch_norm while trainig and testing
        self.is_training = tf.placeholder(tf.bool)

        self.skip_steps = 100
        self.n_steps_to_save = 500

    def get_data(self):
        """
        Define computation graph for processing inputs using Tensorflow Dataset API. 
        """

        # Setup the train generator
        train_generator = DataGenerator(image_path=self.PATH_TO_TRAIN_IMAGES,
                q_path=self.PATH_TO_TRAIN_QUESTIONS,
                a_path=self.PATH_TO_TRAIN_ANSWERS,
                p_path=self.PATH_TO_TRAIN_PICKLE_FILES,
                image_rescale=1, image_horizontal_flip=False, image_target_size=self.IMAGE_SIZE,
                use_num_answers=5, init_code=self.TRAIN_INIT_CODE)

        self.num_train_samples = train_generator.get_num_of_samples()

        validation_generator = DataGenerator(image_path=self.PATH_TO_VALIDATION_IMAGES,
                q_path=self.PATH_TO_VALIDATION_QUESTIONS,
                a_path=self.PATH_TO_VALIDATION_ANSWERS,
                p_path=self.PATH_TO_VALIDATION_PICKLE_FILES,
                image_rescale=1, image_horizontal_flip=False, image_target_size=self.IMAGE_SIZE,
                use_num_answers=5, init_code=self.VAL_INIT_CODE)

        self.num_validation_samples = validation_generator.get_num_of_samples()

        train_data_generator = lambda: train_generator.mini_batch_generator_v2()
        validation_data_generator = lambda: validation_generator.mini_batch_generator_v2()

        train_data = tf.data.Dataset.from_generator(
            generator=train_data_generator,
            output_types=(tf.float32, tf.string, tf.float32),
            output_shapes=(tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)),
        ).batch(self.BATCH_SIZE).prefetch(self.PREFETCH)

        validation_data = tf.data.Dataset.from_generator(
            generator=validation_data_generator,
            output_types=(tf.float32, tf.string, tf.float32),
            output_shapes=(tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)),
        ).batch(self.BATCH_SIZE).prefetch(self.PREFETCH)

        # Load words vocab table for quick lookup
        word_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.PATH_TO_WORD_VOCAB, num_oov_buckets=1)

        iterator = train_data.make_initializable_iterator()

        img, sentence, self.label = iterator.get_next()

        with tf.name_scope('sentence_splitter'):
            sentence = tf.map_fn(lambda x: tf.string_split([x], delimiter=' ').values, sentence, dtype=tf.string)

        with tf.name_scope('word_table_lookup'):
            self.sentence = tf.map_fn(lambda x: tf.cast(word_table.lookup(x), tf.int32), sentence, dtype=tf.int32)

        # Preapre image for a CNN pass
        self.img = tf.reshape(img, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3])

        # Add iterators to the graph
        self.train_init = iterator.make_initializer(train_data)           # Train iterator
        self.validation_init = iterator.make_initializer(validation_data)      # Validation iterator


    def loss(self):
        """
        Define proper loss function for training the neural network.
        """

        with tf.name_scope('loss'):

            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=(self.predictions + 1e-8))

            # Loss is mean of error on all dimensions
            self.loss_val = tf.reduce_mean(cross_entropy_loss, name='loss')

    def optimize(self):
        """
        Define proper optimization method for training the neural network.
        """

        with tf.name_scope('AdamOptimizer'):
            
            all_vars = tf.trainable_variables()
            trainable_vars = [var for var in all_vars if 'resnet' not in var.name]

            print("All Variables: ", all_vars)
            print("Trainable Variables: ", trainable_vars)
            
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.LEARNING_RATE,
                name='AdamOptimizer'
            ).minimize(self.loss_val, global_step=self.g_step, var_list=trainable_vars)

    def eval(self):
        """
        Define the desired evaluation method for the model (Accuracy in this case)
        """

        with tf.name_scope('Accuracy'):
            
            self.softmaxed_preds = tf.nn.softmax(self.predictions)
            self.argmaxed_preds = tf.argmax(self.softmaxed_preds, 1)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.softmaxed_preds, 1), tf.argmax(self.label, 1)), tf.float32))

            self.acc, self.acc_update = tf.metrics.accuracy(labels=tf.argmax(self.label, 1), predictions=tf.argmax(self.softmaxed_preds, 1))

    def summary(self):
        """
        Use Tensorflow mechanisms to display summaries on Tensorboard
        """

        with tf.name_scope('graph_summary'):

            tf.summary.scalar('num_t_preds', self.accuracy)
            # tf.summary.histogram('loss_histogram', self.loss_val)
            tf.summary.scalar('loss_value', self.loss_val)

            self.summary = tf.summary.merge_all()


    def build_model(self):
        """
        Build the core computation graph for processing questions, answers and images.
        """

        # Setup input data pipeline
        self.get_data()

        # Feature extraction for the image
        feature_extractor = FeatureExtractor(keep_prob=1.0,
                                            enable_pre_trained_weights=False,
                                            path_to_pretrained_cnn_weights=self.PATH_TO_PRETRAINED_CNN_WEIGHTS)

        
        # self.pre_trained_cnn_weights_init = feature_extractor.load_trained_model_tensors()

        # Word embeddings
        word_vectorizer = WordVectorizer(batch_size=self.BATCH_SIZE,
                                         glove_file_path=self.PATH_TO_TRAINED_GLOVE,
                                         vocab_file_path=self.PATH_TO_WORD_VOCAB)


        self.embedding_init = word_vectorizer.load_trained_model_tensors()

        # Classifer
        classifier = Classifier(self.NUM_CLASSES)

        # Obtain image feature maps
        # self.image_feature_map = feature_extractor.generate_image_feature_map(self.img)

        self.image_feature_map, self.pre_trained_cnn_weights_init = feature_extractor.generate_image_feature_map_with_resnet(self.img, is_training=self.is_training)

        # # Obtain answer embeddings
        # self.answer_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.answer))

        # # Obtain sentence embeddings
        # self.question_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.question))

        # Obtain sentence embeddings
        self.sentence_glove_vector = tf.layers.flatten(word_vectorizer.generate_sentence_vector(self.sentence))

        # Concatenate image feature map and sentence feature maps
        # self.iqa_vector = tf.concat(values=[self.image_feature_map, self.question_glove_vector, self.answer_glove_vector], axis=1)

        # self.iqa_vector = tf.concat(values=[self.question_glove_vector, self.image_feature_map], axis=1)

        self.iqa_vector = tf.concat(values=[self.sentence_glove_vector, self.image_feature_map], axis=1)

        # Classify the concatenated vector
        self.predictions = classifier.classify_input(self.iqa_vector)

        # Setup loss function
        self.loss()

        # Setup optimizer
        self.optimize()

        # Model accuracy and evaluation
        self.eval()

        # Training/Testing summary
        self.summary()

    def get_model_statistics(self, preds, truths, losses):

        auc_scores = np.zeros(shape=len(self.CLASS_LIST))
        acc_scores = np.zeros(shape=len(self.CLASS_LIST))

        for class_idx in range(len(self.CLASS_LIST)):
            class_name = self.CLASS_LIST[class_idx]
            auc_scores[class_idx] = roc_auc_score(truths[class_name], preds[class_name])
            acc_scores[class_idx] = accuracy_score(truths[class_name], preds[class_name])

        avg_auc = np.mean(auc_scores)
        avg_acc = np.mean(acc_scores)
        avg_loss = np.mean(losses)

        return auc_scores, acc_scores, avg_auc, avg_acc, avg_loss

    def report_model_statistics(self, auc_scores, acc_scores, avg_auc, avg_acc, loss, epoch, start_time, mode):

        if(mode == self.TRAIN_INIT_CODE):
            superb = 'Training'
        else:
            superb = 'Validation'

        print("\n************ {} ************".format(superb))

        print("[CLASS ROC-AUC SCORES]: ")
        for class_idx in range(len(self.CLASS_LIST)):

            class_name = self.CLASS_LIST[class_idx]
            print("{0} -> {1}".format(class_name, auc_scores[class_idx]))

        print("[AVG AUC] at Epoch {0}: {1}".format(epoch, avg_auc))

        print("**************************************")
        print("[CLASS ACCURACY SCORES]: ")
        for class_idx in range(len(self.CLASS_LIST)):

            class_name = self.CLASS_LIST[class_idx]
            print("{0} -> {1}".format(class_name, acc_scores[class_idx]))

        print("[AVG ACC] at Epoch {0}: {1}".format(epoch, avg_acc))
        print("[LOSS] at Epoch {0}: {1}".format(epoch, loss))

        print("[TIMING] Took {0} Seconds...".format(time.time() - start_time))

    def train_one_epoch(self, init, sess, saver, writer, step, epoch):
        """
        Train network one epoch. Display accuracy and loss metrics for based on the number defined in self.skip_steps.
        """

        # Get current system time
        start_time = time.time()

        # Initialize dataset pipeline with train data
        sess.run(init)
        losses = []

        all_predictions = pd.DataFrame(columns=self.CLASS_LIST)
        all_labels = pd.DataFrame(columns=self.CLASS_LIST)

        pbar = tqdm(total=int(self.num_train_samples/self.BATCH_SIZE))
        pbar.set_description("Training Epoch {}".format(epoch))

        # Trace and profile
        run_metadata = tf.RunMetadata()

        try:
            while True:
                # Get accuracy, loss value and optimize the network + summary of validation
                batch_accuracy, step_loss, _, step_summary, preds, truths = sess.run(
                    [self.acc_update, self.loss_val, self.opt, self.summary, self.argmaxed_preds, self.label],
                    feed_dict={self.is_training: True},
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)

                pbar.update(1)
                step += 1
                losses.append(step_loss)

                for pred in preds:
                    all_predictions.loc[len(all_predictions)] = pred

                for truth in truths:
                    all_labels.loc[len(all_labels)] = truth

                # Get step stats
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(self.PATH_TO_TRACE_FILE, 'w')
                trace_file.write(trace.generate_chrome_trace_format())

                # if((step + 1) % self.skip_steps == 0):
                #     print('Loss at step {}: {}'.format(step, step_loss))
                    # writer.add_summary(step_summary, global_step=step)

                # if((step + 1) % self.n_steps_to_save == 0):
                #     save_path = saver.save(sess, self.PATH_TO_MODEL_CHECKPOINTS)
                #     print("Trained weights saved in path: {}".format(save_path))
        except tf.errors.OutOfRangeError:
            pass;


        print("Predictions: ", all_predictions)
        print("Labels: ", all_labels)

        # Get model statistics
        auc_scores, acc_scores, avg_auc, avg_acc, avg_loss = self.get_model_statistics(
                                                                                    all_predictions, all_labels, losses)

        # Report model statistics
        self.report_model_statistics(auc_scores, acc_scores, avg_auc, avg_acc, avg_loss, epoch, start_time, mode=self.TRAIN_INIT_CODE)

        # Write TF summaries
        summary = tf.Summary()
        summary.value.add(tag="Training Loss", simple_value=avg_loss)
        summary.value.add(tag="Training Accuracy", simple_value=avg_acc)
        writer.add_summary(summary, epoch)

        return step

    def validate(self, init, sess, writer, step, epoch):
        """
        Validate the trained model on validation data.
        """
        # Get current system time
        start_time = time.time()

        # Initialize dataset pipeline with validation data
        sess.run(init)
        losses = []
        
        all_predictions = pd.DataFrame(columns=self.CLASS_LIST)
        all_labels = pd.DataFrame(columns=self.CLASS_LIST)

        pbar = tqdm(total=int(self.num_validation_samples/self.BATCH_SIZE))
        pbar.set_description("Validation Epoch {}".format(epoch))
        try:
            while True:
                # Get accuracy and summary of validation
                batch_accuracy, step_loss, step_summary, preds, truths = sess.run([self.acc_update, self.loss_val, self.summary, self.argmaxed_preds, self.label], feed_dict={self.is_training: False})
                pbar.update(1)
                losses.append(step_loss)

                for pred in preds:
                    all_predictions.loc[len(all_predictions)] = pred

                for truth in truths:
                    all_labels.loc[len(all_labels)] = truth

        except tf.errors.OutOfRangeError:
            pass;


        # Get model statistics
        auc_scores, acc_scores, avg_auc, avg_acc, avg_loss = self.get_model_statistics(
                                                                                    all_predictions, all_labels, losses)

        # Report model statistics
        self.report_model_statistics(auc_scores, acc_scores, avg_auc, avg_acc, avg_loss, epoch, start_time, mode=self.VAL_INIT_CODE)

        summary = tf.Summary()
        summary.value.add(tag="Validation Loss", simple_value=avg_loss)
        summary.value.add(tag="Validation Accuracy", simple_value=avg_acc)
        writer.add_summary(summary, epoch)

    def train_and_validate(self, num_epochs):
        """
        Initial setup for saving tensorflow sessions, running training session and validation session.
        """
        
        # Tensorflow writer for graphs and summary saving
        train_writer = tf.summary.FileWriter(self.PATH_TO_TRAIN_VISUALIZATION_GRAPHS, tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.PATH_TO_VALIDATION_VISUALIZATION_GRAPHS)

        # Saving operation (also for resotre)
        saver = tf.train.Saver()

        # GPU Options for allocating a part of GPU Ram :D
        config = tf.ConfigProto(allow_soft_placement=False) # No soft placement on CPU
        config.gpu_options.allow_growth = True

        with tf.Session(config=config).as_default() as sess:

            # Restore model
            if os.path.exists(self.PATH_TO_MODEL_CHECKPOINTS):
                saver.restore(sess, self.PATH_TO_MODEL_CHECKPOINTS)
                print("Model Restored.")
            else:
                # Initialize variables
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            
            # Initialize tables
            sess.run(tf.tables_initializer())

            # Load GloVe embeddings
            sess.run(self.embedding_init)

            # Load pre-trained weights
            self.pre_trained_cnn_weights_init(sess)

            step = self.g_step.eval()

            # Train multiple epochs
            for epoch in range(num_epochs):
                
                # Train one epoch
                step = self.train_one_epoch(
                    init=self.train_init,
                    sess=sess,
                    saver=saver,
                    writer=train_writer,
                    step=step,
                    epoch=epoch
                )

                # Validate once
                self.validate(
                    init=self.validation_init,
                    sess=sess,
                    writer=validation_writer,
                    step=step,
                    epoch=epoch
                )