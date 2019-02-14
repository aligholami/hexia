import argparse
from options import VQAOptions


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()

    # parse paths
    parser.add_argument('--qa_path', type=str, default='../../data/qa')
    parser.add_argument('--train_images_path', type=str, default='../../data/train/images/')
    parser.add_argument('--val_images_path', type=str, default='../../data/validation/images/')
    parser.add_argument('--test_images_path', type=str, default='../../data/test/images/')
    parser.add_argument('--word2vec_embeddings_path', type=str, default='../../models/glove-embeddings/glove.6B.50d.txt')

    # parse intermediate data
    parser.add_argument('--images_features_path', type=str, default='../../models/pretrained-cnn-weights/resnet/r101_weights.h5')
    parser.add_argument('--images_preprocess_batch_size', type=int, default=64)
    parser.add_argument('--qa_vocab_path', type=str, default='../../models/qa-vocab/vocab.json')
    parser.add_argument('--word2vec_processed_embeddings_path', type=str, default='../../models/glove-embeddings/glove.6B.50d.dat')
    parser.add_argument('--word2vec_words_only_path', type=str, default='../../models/glove-embeddings/glove.6B.50_words.pkl')
    parser.add_argument('--word2vec_ids_only_path', type=str, default='../../models/glove-embeddings/glove.6B.50_idx.pkl')

    parser.add_argument('--model_outputs_path', type=str, default='../../model-output/OpenEnded_mscoco_val2014_org_results')

    # specify desired options
    vqa_options = VQAOptions()