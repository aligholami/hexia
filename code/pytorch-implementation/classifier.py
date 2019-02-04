import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Perform forward pass for the classifier.
        :param x: Features to classify.
        :return:  Predictions without Softmax.
        """

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
