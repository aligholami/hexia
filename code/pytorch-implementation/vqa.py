import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .classifier import Classifier


class VQA(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(VQA, self).__init__()

        self.feature_map = FeatureExtractor()
        self.classifier = Classifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    def forward(self, x):
        """
        Implement forward pass for the VQA model.
        :param x: Input data (Image, Question, Answer, Label)
        :return: VQA Predictions.
        """

        fm = self.feature_map(x)
        predictions = self.classifier(fm)

        return predictions


