from image_generator import ImageGenerator


class VQA_SAN:

    PATH_TO_TRAIN_IMAGES = '../data/train'
    BATCH_SIZE = 10
    
    def __init__(self):

        pass;

    def get_data(self):

        # Setup Image Generator
        train_image_generator = ImageGenerator(path_to_generate=PATH_TO_TRAIN_IMAGES, rescale=1, horizontal_flip=False, target_size=(150, 150))
        train_image_generator = train_image_generator.image_mb_generator(batch_size=BATCH_SIZE)
        train_images_names = train_image_generator.filenames
        