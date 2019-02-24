import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
import bcolz
import pickle
import json
from tqdm import tqdm
import config
import data

train_iters = 0
val_iters = 0

def reload_dataset_vocab():
    """
    Reloads VQA V2 dataset vocabulary into memory to be used by models (Text Processor).
    :return: A dictionary of mappings from words to ids for both questions and answers.
    """

    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)

    # Skip integrity test

    # vocab
    vocab = vocab_json
    token_to_index = vocab['question']
    answer_to_index = vocab['answer']

    return token_to_index, answer_to_index


def reload_glove_embeddings():
    """
    Reload the GloVe embeddings after running the prepare_vocab.py file. This will be used in the model.
    :return: A dictionary of mappings from words to vectors
    """
    vectors = bcolz.open(config.glove_processed_vectors)[:]
    words = pickle.load(open(config.glove_words, 'rb'))
    word2idx = pickle.load(open(config.glove_ids, 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    return glove


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers

    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10

    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5 ** (float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def prepare_data_loaders():
    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    return train_loader, val_loader


def save_for_vqa_evaluation(anws, ids, qids, epoch):
    
    # Load vocab json to obtain inverse list
    idx2word = {}

    with open(config.vocabulary_path) as vocab_json:
        word2idx = json.load(vocab_json)
        a_word2idx = word2idx['answer']

        for word, id in a_word2idx.items():
            idx2word[id] = word

    evaluation_list = []
    for i, id in enumerate(ids):
        evaluation_list.append({
            "answer": "{}".format(idx2word.get(anws[i].item())),
            "question_id": qids[i].item()
        })
    
    pth = config.eval_results_path
    pth += '_' + str(epoch)
    pth += '.json'
    with open(pth, 'w') as eFile:
        json.dump(evaluation_list, eFile)

def run(net, loader, optimizer, tracker, writer, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []
        qids = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    for qid, v, q, a, idx, q_lens in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)

        # used for sequence padding and packing
        q_lens = Variable(q_lens.cuda(async=True), **var_params)

        out = net(v, q, q_lens)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = batch_accuracy(out.data, a.data).cpu()

        if train:
            global train_iters
            update_learning_rate(optimizer, train_iters)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iters += 1
            # Write loss and accuracy to TensorboardX
            writer.add_scalar('/loss', loss.data.item(), train_iters)
            writer.add_scalar('/accuracy', acc.mean(), train_iters)

        else:
            global val_iters
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            qids.append(qid.view(-1).clone())
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())
            val_iters += 1
            
            # Write loss and accuracy to TensorboardX
            writer.add_scalar('/loss', loss.data.item(), val_iters)
            writer.add_scalar('/accuracy', acc.mean(), val_iters)

        loss_tracker.append(loss.data.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        qids = list(torch.cat(qids, dim=0))

        # Save results for vqa evaluation
        save_for_vqa_evaluation(answ, idxs, qids, epoch)

        return answ, accs, idxs


def path_for(train=False, val=False, test=False, question=False, answer=False):
    assert train + val + test == 1
    assert question + answer == 1
    assert not (
            test and answer), 'loading answers from test split not supported'  # if you want to eval on test, you need to implement loading of a VQA Dataset without given answers yourself
    if train:
        split = 'train2014'
    elif val:
        split = 'val2014'
    else:
        split = 'test2015'
    if question:
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        fmt = 'v2_{1}_{2}_annotations.json'
    s = fmt.format(config.task, config.dataset, split)
    return os.path.join(config.qa_path, s)


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        """ Storage of data points that updates the given monitors """

        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value


def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
