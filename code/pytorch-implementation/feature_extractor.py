import torch.nn as nn


class FeatureExtractor(nn.Module):

    class Flatten(nn.Module):
        def forward(self, x):
            """
            Flatten input x.
            :param x: An n-dimensional matrix.
            :return: Flattened x.
            """
            return x.view(x.size(0), -1)

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            self.Flatten()
        )

    def forward(self, x):
        """
        Peform forward pass for the convolutonal neural network.
        :param x: Image input to the feature extractor.
        :return: Image feature map (flattened)
        """

        out = self.conv1(x)
        out = self.conv2(out)

        return out  # Flattened feature maps
