import tensorflow as tf
from data_generator import DataGenerator
import utils

class VQA_SAN:

    PATH_TO_TRAIN_IMAGES = '../data/train/images/full-image-dir'
    PATH_TO_TRAIN_QUESTIONS = '../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    PATH_TO_TRAIN_ANSWERS = '../data/train/answers/v2_mscoco_train2014_annotations.json'
    BATCH_SIZE = 1
    
    def __init__(self):

        pass;

    def get_data(self):

        # Setup the generator
        train_generator = DataGenerator(image_path=self.PATH_TO_TRAIN_IMAGES,
                        q_path=self.PATH_TO_TRAIN_QUESTIONS,
                        a_path=self.PATH_TO_TRAIN_ANSWERS, 
                        image_rescale=1, image_horizontal_flip=False, image_target_size=(150, 150))
        
        train_generator = train_generator.mini_batch_generator(batch_size=self.BATCH_SIZE)
        
        # # Connect to Tensorflow dataset API
        # train_data = tf.data.Dataset.from_generator(
        #     generator=train_generator,
        #     output_types=
        # )


        




        return train_generator

