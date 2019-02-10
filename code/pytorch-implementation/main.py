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






    # specify desired options
    vqa_options = VQAOptions()