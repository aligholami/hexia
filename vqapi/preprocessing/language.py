import itertools
from collections import Counter

class Language:

    def __init__(self):
        pass

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

    