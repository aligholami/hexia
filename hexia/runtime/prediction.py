import torch
from hexia.tests import config
import json

class VQAPrediction:

    def init(self, path_to_best_vqa_weights, raw_model):
        self.path_to_best_vqa_weights = path_to_best_vqa_weights
        self.raw_model = raw_model

    def load_model_for_inference(self):
        """
            Loads a given raw model and the trained weights into the memory.
        """
        self.loaded_model = self.raw_model
        self.loaded_model.load_state_dict(torch.load(self.path_to_best_vqa_weights))
        self.loaded_model.eval()

        # Prepare idx to word vocab (Maps each of the 3000 answer idx to the natural language answer)
        # This should be right here to speed up the answering process (Memory Preloading)
        self.idx2word = self.prepare_idx_to_word_vocab()

    def prepare_idx_to_word_vocab(self):

        # Load vocab json to obtain inverse list
        idx2word = {}

        with open(config.vocabulary_path) as vocab_json:
            word2idx = json.load(vocab_json)
            a_word2idx = word2idx['answer']

            for word, id in a_word2idx.items():
                idx2word[id] = word

        return idx2word

    def get_natural_answers(self, anws):
        """
            Returns a list of natural language answers for the input answer vectors.
        """
        natural_answers = []

        for i, answer_id in enumerate(anws):
            natural_answers.append(self.idx2word.get(anws[i].item()))

        return natural_answers

    def predict(self, list_of_iq_pairs):

        # List of all answers to the questions
        answers = []

        # Get the corresponding question ids out of vocab

        # Predict the answers
        for idx, i, q in enumerate(list_of_iq_pairs):

            # Assert required here
            # Predict one pair at a time
            q_len = len(q.split(' '))
            answer = self.loaded_model(i, q, len)
            answers.append(answer)

        natural_answers = self.get_natural_answers(answers)

        return natural_answers