import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
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
            rnn_hidden_size=config.rnn_hidden_size
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_lens):

        q = self.text(q, list(q_len.data))

        # Flatten visual features
        v = v.view(v.size(0), -1)

        # Get the mean of question embeddings along axis 1
        q = torch.mean(q, dim=1)

        # Flatten question features
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
    def __init__(self, embedding_tokens, embedding_features, rnn_hidden_size):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.recurrent_layer = nn.RNN(input_size=embedding_features, hidden_size=rnn_hidden_size, num_layers=1)
        self.tanh = nn.Tanh()
        init.xavier_uniform(self.embedding.weight)

    def forward(self, q, q_lens):
        embedded = self.embedding(q)

        # apply non-linearity
        tanhed = self.tanh(embedded)

        # pack sequence
        padded = pack_padded_sequence(tanhed, q_lens)

        # apply rnn
        output, hn = self.recurrent_layer(padded)

        return output
