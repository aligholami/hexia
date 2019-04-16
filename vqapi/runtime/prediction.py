import torch

class VQAPrediction:

    def __init__(self, path_to_saved_vqa_model, raw_model):
        self.path_to_saved_vqa_model = path_to_saved_vqa_model
        self.raw_model = raw_model

    def load_model_for_inference(self):
        """
            Loads a given raw model and the trained weights into the memory.
        """
        self.loaded_model = self.raw_model
        self.loaded_model.load_state_dict(torch.load(self.path_to_saved_vqa_model))
        self.loaded_model.eval()

    def predict(self, list_of_iql_pairs):

        for idx, i, q, q_len in enumerate(list_of_iq_pairs):
            answer = self.loaded_model(i, q, q_len)