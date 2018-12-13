from image_generator import ImageGenerator
from text_generator import TextGenerator

class DataGenerator:

    def __init__(self, image_path, q_path, a_path, image_rescale, image_horizontal_flip, image_target_size):
        self.image_path = image_path
        self.q_path = q_path
        self.a_path = a_path
        self.image_rescale = image_rescale
        self.image_horizontal_flip = image_horizontal_flip
        self.image_target_size = image_target_size

    def mini_batch_generator(self, batch_size):
        
        # Generate a batch of images
        # for each image in the batch generate 
        return 