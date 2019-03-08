import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config


class Net(nn.Module):

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()

        # Get number of visual and embedding features
        vision_features = config.output_size * config.output_size * config.output_features
        embedding_features = config.embedding_features

        self.classifier = Classifier(
            in_features=vision_features + config.lstm_hidden_size,
            mid_features=config.mid_features,
            out_features=config.max_answers,
        )

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=config.embedding_features,
            lstm_hidden_size=config.lstm_hidden_size
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_lens):

        q = self.text(q, list(q_lens.data))

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
    def __init__(self, embedding_tokens, embedding_features, lstm_hidden_size):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.recurrent_layer = nn.LSTM(input_size=embedding_features, hidden_size=lstm_hidden_size, num_layers=1)
        self.tanh = nn.Tanh()
        init.xavier_uniform(self.embedding.weight)

    def forward(self, q, q_lens):
        embedded = self.embedding(q)

        # apply non-linearity
        tanhed = self.tanh(embedded)

        # pack sequence
        packed = pack_padded_sequence(tanhed, q_lens, batch_first=True)
    
        # apply rnn
        output, (hn, cn) = self.recurrent_layer(packed)

        # re-pad sequence 
        padded = pad_packed_sequence(output, batch_first=True)[0]

        # re-order
        padded = padded.contiguous()

        return padded

class AttentionMechanism(nn.Module):

    def __init__(self):
        super(AttentionMechanism, self).__init__()      # register attention class

        self.image_linear_transform = nn.Linear(in_features= , out_features=, bias=True)
        self.quetion_linear_transform = nn.Linear(in_features=, out_features = bias=True)
        self.merged_linear_transform = nn.Linear(in_features=, out_features, bias=True)

    def forward(v_q, v_i):
        
        # divide v_i into regions

        # apply linear transformations in the question and the image 
        h_q = self.quetion_linear_transform(v_q)
        h_i = self.image_linear_transform(v_i)

        # merge two vectors by adding columns of h_i to h_q
        h_merged = something

        # apply linear transformation on h_merged
        attention_map = self.merged_linear_transform(h_merged)

        # apply softmax to get attention distribution on image regions
        attention_distributions = F.softmax(attention_map, dim=1)

        # multiply each score with corresponding image region

        # merge attended images with question vector and return this vector

        




        


