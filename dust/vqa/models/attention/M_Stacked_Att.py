import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from dust.tests import config


class Net(nn.Module):

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()

        # Get number of visual and embedding features
        vision_features = config.output_size * config.output_size * config.output_features
        embedding_features = config.embedding_features

        self.classifier = Classifier(
            in_features=config.lstm_hidden_size,
            mid_features=config.mid_features,
            out_features=config.max_answers,
        )

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=config.embedding_features,
            lstm_hidden_size=config.lstm_hidden_size
        )

        self.attention_pass = AttentionMechanism()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_lens):

        q = self.text(q, list(q_lens.data))

        # perform attention
        attented_v = self.attention_pass(v, q)

        # Flatten visual features
        attented_v = attented_v.view(attented_v.size(0), -1)

        # Normalzie visual features
        attented_v = attented_v / (attented_v.norm(p=2, dim=1, keepdim=True).expand_as(attented_v) + 1e-8)

        # Get the answer predictions
        answer = self.classifier(attented_v)

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

        padded = pad_sequence(tanhed, batch_first=True)

        # pack sequence
        packed = pack_padded_sequence(padded, q_lens, batch_first=True)

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

        self.l1_v_i = nn.Linear(config.output_features, config.lstm_hidden_size, bias=True)
        self.l1_tanh = nn.Tanh()
        self.l2_v_i = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size)
        self.l2_v_q = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size, bias=True)
        self.l2_tanh = nn.Tanh()
        self.l3_h_a = nn.Linear(1, config.lstm_hidden_size, bias=True)
        self.l3_softmax = nn.Softmax(dim=1)

    def forward(self, v_i, v_q):

        # convert image feature map to image regions feature matrix
        v_i = v_i.view(v_i.size(0), v_i.size(1), -1)

        # swap feature map dimensions with attention regions dimension
        v_i = v_i.permute(0, 2, 1)

        # apply a linear transformation on v_i to make its rows the same size as q_lens
        v_i = self.l1_tanh(self.l1_v_i(v_i))

        # find mean of the question
        v_q = torch.mean(v_q, dim=1)
        v_q = torch.squeeze(v_q)    # remove dim with number 1

        u = v_q
        q = v_q

        num_attention_layers = 2
        for i in range(num_attention_layers):

            # Define new weights
            params = {
                'v_i_weights': nn.Parameter(torch.zeros([config.lstm_hidden_size, config.lstm_hidden_size]).cuda()),
                'u_weights': nn.Parameter(torch.zeros([config.lstm_hidden_size, config.lstm_hidden_size]).cuda()),
                'u_biases': nn.Parameter(torch.ones([config.lstm_hidden_size]).cuda())
            }

            # Xaiver initialize the weights
            torch.nn.init.xavier_uniform(params['v_i_weights'])
            torch.nn.init.xavier_uniform(params['u_weights'])

            v_i_t = v_i
            v_i_t = F.linear(v_i_t, params['v_i_weights'])
            v_q_t = F.linear(u, params['u_weights'], params['u_biases'])

            h_a = F.tanh(v_i_t.add(v_q_t[:, None, :]))
            p_i = F.softmax(h_a)

            # multiply distribution to the image regions features
            v_i_hat = torch.mul(p_i, v_i)

            # v_i_hat is a matrix of size (batch_size, config.lstm_hidden_size, config.output_size^2)
            v_i_hat = torch.sum(v_i_hat, dim=1)

            u = u + v_i_hat

        return u






