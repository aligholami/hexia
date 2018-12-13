from keras.preprocessing.image import ImageDataGenerator

class ImageGenerator:

    def __init__(self, path_to_generate, rescale, horizontal_flip, target_size):

        self.path_to_generate = path_to_generate
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.target_size = target_size

    def image_mb_generator(self, batch_size):

        img_datagen = ImageDataGenerator(rescale=self.rescale, horizontal_flip=self.horizontal_flip)
        data_generator = img_datagen.flow_from_directory(self.path_to_generate, batch_size=batch_size, target_size=self.target_size)
        
        return data_generator

    
        