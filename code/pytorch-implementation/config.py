# paths
qa_path = '../../data/qa'  # directory containing the question and annotation jsons
train_path = '../../data/train/images/'  # directory of training images
val_path = '../../data/validation/images/'  # directory of validation images
test_path = '../../data/test/images/'  # directory of test images
vocabulary_path = '../../models/qa-vocab/vocab.json'  # path where the used vocabularies for question and answers are
visualization_dir = '../../visualization/'
glove_embeddings = '../../models/glove-embeddings/glove.6B.50d.txt'
glove_processed_vectors = '../../models/glove-embeddings/glove.6B.50d.dat'
glove_words = '../../models/glove-embeddings/glove.6B.50_words.pkl'
glove_ids = '../../models/glove-embeddings/glove.6B.50_idx.pkl'
eval_results_path = '../../eval-results/OpenEnded_mscoco_val2014_org_results'

# saved to
preprocessed_path = '../../models/pretrained-cnn-weights/resnet/r101_weights.h5'
task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 128  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
embedding_features = 50
mid_features = 8192

# training config
num_epochs = 25
batch_size = 512
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000

# model config
rnn_hidden_size = 300
