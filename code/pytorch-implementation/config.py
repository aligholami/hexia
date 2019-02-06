# paths
qa_path = '../../data/qa'  # directory containing the question and annotation jsons
train_path = '../../data/train/images/'  # directory of training images
val_path = '../../data/validation/images/'  # directory of validation images
test_path = '../../data/test/images/'  # directory of test images
vocabulary_path = '../../models/qa-vocab/vocab.json'  # path where the used vocabularies for question and answers are
# saved to
preprocessed_path = '../../models/pretrained-cnn-weights/resnet/r18_weights.h5'
task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 128  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 512  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
embedding_features = 300
mid_features = 8192

# training config
num_epochs = 30
batch_size = 64
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
