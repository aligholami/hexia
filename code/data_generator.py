from image_generator import ImageGenerator
# from text_generator import TextGenerator

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

        train_image_generator = ImageGenerator(path_to_generate=self.image_path, rescale=1, horizontal_flip=False, target_size=(150, 150))
        train_image_generator = train_image_generator.image_mb_generator(batch_size=batch_size)


        for i in train_image_generator:
            idx = (train_image_generator.batch_index - 1) * train_image_generator.batch_size
            target_file = train_image_generator.filenames[idx: idx + train_image_generator.batch_size]

            yield target_file