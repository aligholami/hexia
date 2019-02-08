import numpy as np
import bcolz
import pickle
import json
from collections import Counter
import itertools
import config
import data
import utils


def prepare_glove_embeddings(dims=50):
    g_words = []
    g_idx = 0
    g_word2idx = {}
    g_vectors = bcolz.carray(np.array(1), rootdir=config.glove_processed_vectors, mode='w')

    # Get glove weights and vocab
    with open(config.glove_embeddings, 'rb') as gfile:
        for line in gfile:
            line = line.decode('utf-8').split()
            word = line[0]
            g_words.append(word)
            g_word2idx[word] = g_idx
            g_idx += 1
            wvec = np.array(line[1:]).astype(np.float)
            g_vectors.append(wvec)

    g_vectors = bcolz.carray(g_vectors[1:].reshape((400000, dims)), rootdir=config.glove_processed_vectors, mode='w')
    g_vectors.flush()

    # Save GloVe words and dicts
    pickle.dump(g_words, open(config.glove_words, 'wb'))
    pickle.dump(g_word2idx, open(config.glove_ids, 'wb'))


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """

    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}

    return vocab


def main():
    questions = utils.path_for(train=True, question=True)
    answers = utils.path_for(train=True, answer=True)

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data.prepare_questions(questions)
    answers = data.prepare_answers(answers)

    # Prepare GloVe embeddings
    prepare_glove_embeddings(dims=50)

    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
