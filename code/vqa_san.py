import tensorflow as tf
from feature_extractor import FeatureExtractor
from word_vectorizer import WordVectorizer
from classifier import Classifier
from data_generator import DataGenerator
from utils import load_embedding_from_disks

class VQA_SAN:

    PATH_TO_TRAIN_IMAGES = '../data/train/images/full-image-dir'
    PATH_TO_TRAIN_QUESTIONS = '../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    PATH_TO_TRAIN_ANSWERS = '../data/train/answers/v2_mscoco_train2014_annotations.json'
    PATH_TO_TRAINED_GLOVE = '../models/GloVe/glovefile.txt'
    BATCH_SIZE = 32
    PREFETCH = 32
    
    
    def __init__(self):
        pass;

    def get_data(self):

        # Setup the generator
        train_generator = DataGenerator(image_path=self.PATH_TO_TRAIN_IMAGES,
                        q_path=self.PATH_TO_TRAIN_QUESTIONS,
                        a_path=self.PATH_TO_TRAIN_ANSWERS, 
                        image_rescale=1, image_horizontal_flip=False, image_target_size=(150, 150))
        
        train_generator = train_generator.mini_batch_generator(batch_size=self.BATCH_SIZE)
        
        train_data = tf.data.Dataset.from_generator(
            generator=train_generator,
            output_types=(tf.float32, tf.string, tf.string),
            output_shapes=(tf.TensorShape[None], tf.TensorShape[None], tf.TensorShape[None]),
        ).batch(self.BATCH_SIZE).prefetch(self.PREFETCH)

        iterator = train_data.make_initializable_iterator()

        self.img, self.question, self.answer = iterator.get_next()
        

    def build_model(self):

        # Feature extraction for the image
        feature_extractor = FeatureExtractor(flatten=True)
        
        # Word embeddings
        word_vectorizer = WordVectorizer(batch_size=self.BATCH_SIZE,
                                         glove_file_path=self.PATH_TO_TRAINED_GLOVE)

        # Classifer
        classifier = Classifier(num_classes)

        # Obtain image feature maps
        image_feature_map = feature_extractor.generate_image_feature_map(self.img)

        # Obtain word embeddings 
        word_glove_vector = word_vectorizer.generate_word_vector(self.answer)

        # Obtain sentence embeddings
        # sentence_glove_vector = sentence_vectorizer.generate_sentence_vector(self.question)
        # *******************************************************************

        # Concatenate image feature map and sentence feature map
        image_question_vector = tf.concat(concat_dim=0, values=[image_feature_map, sentence_glove_vector], name='feature_merger')

        predictions = classifier.classify_input(image_question_vector)










