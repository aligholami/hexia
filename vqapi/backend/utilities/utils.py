import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
import bcolz
import pickle
import json
from tqdm import tqdm
from vqapi.backend.dataset.data import DataLoadUtils

train_iters = 0
val_iters = 0

class VQAUtils:

    def __init__(self):
        pass

    @staticmethod
    def reload_dataset_vocab(vocab_path):
        """
        Reloads VQA V2 dataset vocabulary into memory to be used by models (Text Processor).
        :return: A dictionary of mappings from words to ids for both questions and answers.
        """

        with open(vocab_path, 'r') as fd:
            vocab_json = json.load(fd)

        # Skip integrity test

        # vocab
        vocab = vocab_json
        token_to_index = vocab['question']
        answer_to_index = vocab['answer']

        return token_to_index, answer_to_index

    @staticmethod
    def reload_glove_embeddings(glove_vectors, glove_words, glove_ids):
        """
        Reload the GloVe embeddings after running the prepare_vocab.py file. This will be used in the model.
        :return: A dictionary of mappings from words to vectors
        """
        vectors = bcolz.open(glove_vectors)[:]
        words = pickle.load(open(glove_words, 'rb'))
        word2idx = pickle.load(open(glove_ids, 'rb'))
        glove = {w: vectors[word2idx[w]] for w in words}

        return glove

    @staticmethod
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

    @staticmethod
    def update_learning_rate(optimizer, iteration, initial_lr, lr_halflife):
        lr = initial_lr * 0.5 ** (float(iteration) / lr_halflife)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def prepare_data_loaders(path_to_feature_maps, batch_size, num_workers):

        data_utils = DataLoadUtils(path_to_feature_maps=path_to_feature_maps, batch_size=batch_size, num_worker_threads=num_workers)

        train_loader = data_utils.get_loader(train=True)
        val_loader = data_utils.get_loader(val=True)

        return train_loader, val_loader

    @staticmethod
    def save_for_vqa_evaluation(anws, ids, qids, vocabulary_path, eval_results_path):

        # Load vocab json to obtain inverse list
        idx2word = {}

        with open(vocabulary_path) as vocab_json:
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

        pth = eval_results_path

        with open(pth, 'w') as eFile:
            json.dump(evaluation_list, eFile)

    @staticmethod
    def path_for(train=False, val=False, test=False, question=False, answer=False, task, dataset, qa_path):
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
        s = fmt.format(task, dataset, split)
        return os.path.join(qa_path, s)

    @staticmethod
    def get_transform(target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Scale(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
