import torch
import torch.nn as nn
import torch.nn.init as init
import config


class Net(nn.Module):

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()

        # Get number of visual and embedding features
        vision_features = config.output_size * config.output_size * config.output_features
        embedding_features = config.embedding_features

        self.classifier = Classifier(
            in_features=vision_features + embedding_features,
            mid_features=config.mid_features,
            out_features=config.max_answers,
        )

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=config.embedding_features,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):

        q = self.text(q, list(q_len.data))

        # Flatten visual features
        v = v.view(v.size(0), -1)

        # Get the mean of question embeddings along axis 1
        q = torch.mean(q, dim=1)

        # Flatten question embeddings
        q = q.view(q.size(0), -1)

        # Normalzie visual features
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # Concatenate visual features and embeddings
        combined = torch.cat([v, q], dim=1)

        # Get the answer predictions
        answer = self.classifier(combined)

        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features):
        super(Classifier, self).__init__()
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)

        init.xavier_uniform(self.embedding.weight)

    def forward(self, q, q_len):
        embedded = self.embedding(q)

        return embedded
