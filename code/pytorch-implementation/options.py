
class VQAOptions:

    def __init__(self, args):

        # Fundamental data
        self.qa_path = args.qa_path
        self.train_images_path = args.train_images_path
        self.val_images_path = args.val_images_path
        self.test_images_path = args.test_images_path
        self.word2vec_embeddings = args.word2vec_embeddings

        # Intermediate data
        self.images_features_path = args.images_features_path
        self.images_preprocess_batch_size = args.images_preprocess_batch_size

        self.qa_vocab_path = args.qa_vocab_path
        self.word2vec_processed_embeddings = args.word2vec_processed_embeddings
        self.word2vec_words_only = args.word2vec_words_only
        self.word2vec_ids_only = args.word2vec_ids_only

        # Evaluation and visualization paths
        self.ready_to_eval_results_path = args.ready_to_eval_results_path

        # Core parameters
        self.vqa_task = args.vqa_task
        self.vqa_dataset = args.vqa_dataset
        self.input_image_size = args.input_image_size
        self.feature_map_spatial_size = args.feature_map_spatial_size
        self.feature_map_depth_size = args.feature_map_depth_size
        self.word2vec_embedding_size = args.word2vec_embedding_size
        self.mlp_num_hidden_units = args.mlp_num_hidden_units

        # Training and evaluation parameters
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.initial_learning_rate = args.initial_learning_rate
        self.learning_rate_half_life = args.learning_rate_half_life
        self.num_data_workers = args.num_data_workers
        self.num_max_answers = args.num_max_answers
