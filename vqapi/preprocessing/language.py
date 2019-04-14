import numpy as np
import itertools
import json
import pickle
import bcolz
from collections import Counter
from vqapi.backend.utilities import utils
from vqapi.backend.dataset import data

class Language:

    def __init__(self, max_answers, save_vocab_to):
        self.max_answers = max_answers
        self.save_vocab_to = save_vocab_to

    def extract_vocab(self, iterable, top_k=None, start=0):
        """ 
            Turns an iterable of list of tokens into a vocabulary.
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

    def initiate_vocab_extraction(self):
        """
            Extract vocabs and save them to the proper directory.
        """
        questions = utils.path_for(train=True, question=True)
        answers = utils.path_for(train=True, answer=True)

        with open(questions, 'r') as fd:
            questions = json.load(fd)
        with open(answers, 'r') as fd:
            answers = json.load(fd)

        questions = data.prepare_questions(questions)
        answers = data.prepare_answers(answers)

        question_vocab = self.extract_vocab(questions, start=1)
        answer_vocab = self.extract_vocab(answers, top_k=self.max_answers)

        vocabs = {
            'question': question_vocab,
            'answer': answer_vocab
        }

        with open(self.save_vocab_to, 'w') as fd:
            json.dump(vocabs, fd)

    def extract_glove_embeddings(self, path_to_pretrained_embeddings, save_vectors_to, save_words_to, save_ids_to):
        """
            Extract glove embeddings and save them to the proper directory.
        """
        g_words = []
        g_idx = 0
        g_word2idx = {}
        g_vectors = bcolz.carray(np.array(1), rootdir=save_vectors_to, mode='w')

        # Get glove weights and vocab
        with open(path_to_pretrained_embeddings, 'rb') as gfile:
            for line in gfile:
                line = line.decode('utf-8').split()
                word = line[0]
                g_words.append(word)
                g_word2idx[word] = g_idx
                g_idx += 1
                wvec = np.array(line[1:]).astype(np.float)
                g_vectors.append(wvec)

        g_vectors = bcolz.carray(g_vectors[1:].reshape((400000, dims)), rootdir=save_vectors_to, mode='w')
        g_vectors.flush()

        # Save GloVe words and dicts
        pickle.dump(g_words, open(save_words_to, 'wb'))
        pickle.dump(g_word2idx, open(save_ids_to, 'wb'))

        


    