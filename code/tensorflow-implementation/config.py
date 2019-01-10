from configparser import SafeConfigParser


vqa_config = SafeConfigParser()
vqa_config.read('configuration.ini')

# Configuration for Convolutional Neural Network
vqa_config.add_section('INPUT')
vqa_config.add_section('CNN')
vqa_config.add_section("Classifier")
vqa_config.add_section('Training')

# Add configuration setup for Input
vqa_config.set('INPUT', 'IMAGE_DIMENSION', '64')
vqa_config.set('INPUT', 'PATH_TO_TRAIN_IMAGES', '../data/train/images/full-image-dir')
vqa_config.set('INPUT', 'PATH_TO_TRAIN_QUESTIONS', '../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json')
vqa_config.set('INPUT', 'PATH_TO_TRAIN_ANSWERS', '../data/train/answers/v2_mscoco_train2014_annotations.json')
vqa_config.set('INPUT', 'PATH_TO_VALIDATION_IMAGES', '../data/validation/images/full-image-dir')
vqa_config.set('INPUT', 'PATH_TO_VALIDATION_QUESTIONS', '../data/validation/questions/v2_OpenEnded_mscoco_val2014_questions.json')
vqa_config.set('INPUT', 'PATH_TO_VALIDATION_ANSWERS', '../data/validation/answers/v2_mscoco_val2014_annotations.json')
vqa_config.set('INPUT', 'PATH_TO_TRAINED_GLOVE', '../models/GloVe/glove.6B.50d.txt')
vqa_config.set('INPUT', 'PATH_TO_WORD_VOCAB', '../models/GloVe/vocab_only.txt')
vqa_config.set('INPUT', 'PATH_TO_VISUALIZATION_GRAPHS', '../visualization/')
vqa_config.set('INPUT', 'PATH_TO_MODEL_CHECKPOINTS', '../models/checkpoints')

# Add configuration setup for CNN operations
vqa_config.set('CNN', 'CONV1_NUM_FILTERS', '10')
vqa_config.set('CNN', 'CONV1_FILTER_SIZE', '3')
vqa_config.set('CNN', 'CONV2_NUM_FILTERS', '15')
vqa_config.set('CNN', 'CONV2_FILTER_SIZE', '3')
vqa_config.set('CNN', 'CONV3_NUM_FILTERS', '20')
vqa_config.set('CNN', 'CONV3_FILTER_SIZE', '3')

# Add configuration setup for classifer
vqa_config.set('Classifier', 'NUM_CLASSES', '3')

# Add configuration setup for training mechanism
vqa_config.set('Training', 'BATCH_SIZE', '32')
vqa_config.set('Training', 'LEARNING_RATE', '1e-4')

with open('configuration.ini', 'w') as con_file:
    vqa_config.write(con_file)