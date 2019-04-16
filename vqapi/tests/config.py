# paths
qa_path = '../../data/qa'  # directory containing the question and annotation jsons
train_path = '../../VQA/data/images/mscoco/train2014/'  # directory of training images
val_path = '../../VQA/data/images/mscoco/val2014/'  # directory of validation images
test_path = '../../VQA/data/test/images/'  # directory of test images
vocabulary_path = '../../models/qa-vocab/vocab.json'  # path where the used vocabularies for question and answers are
visualization_dir = '../../visualization/'
glove_embeddings = '../../models/glove-embeddings/glove.6B.50d.txt'
glove_processed_vectors = '../../models/glove-embeddings/glove.6B.50d.dat'
glove_words = '../../models/glove-embeddings/glove.6B.50_words.pkl'
glove_ids = '../../models/glove-embeddings/glove.6B.50_idx.pkl'
best_vqa_answers_to_eval = '../../eval-results/OpenEnded_mscoco_val2014_org_results.json'

# saved to
preprocessed_path = './prep/pretrained-cnn-weights/resnet/r101_weights.h5'
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
best_vqa_weights_path = './prep/saved-models/bestVqaModel.pth'
latest_vqa_results_path = './prep/saved-models/latestVqaModel.pth'

# model config
rnn_hidden_size = 300
lstm_hidden_size = 300

# attention config
h_a_q_size = 100
h_a_i_rows = 32
h_a_i_cols = 32
num_attention_regions = 4

