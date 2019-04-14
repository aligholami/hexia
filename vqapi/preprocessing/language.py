import itertools
import json
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
        questions = utils.path_for(train=True, question=True)
        answers = utils.path_for(train=True, question=True)

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

        


    