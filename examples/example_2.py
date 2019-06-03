from dustorch.preprocessing.language import Language
from dustorch.tests import config

language_preprocessor = Language(
    max_answers=config.max_answers,
    save_vocab_to=config.vocabulary_path
)
language_preprocessor.initiate_vocab_extraction()

# Remove this line if you want to use Word2Vec embeddings
language_preprocessor.extract_glove_embeddings(
    dims=50,
    path_to_pretrained_embeddings=config.glove_embeddings,
    save_vectors_to=config.glove_processed_vectors,
    save_words_to=config.glove_words,
    save_ids_to=config.glove_ids
)