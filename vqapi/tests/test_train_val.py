import unittest
from vqapi.runtime.train_val import TrainValidation

import torch
import torch.nn as nn
import torch.optim as optim
import config
from vqapi.backend.utilities import utils
from tensorboardX import SummaryWriter
from models import M_ResNet101_randw2v_NoAtt_LSTM as model

class TrainValidationTest(unittest.TestCase):

    def __init__(self):
        pass
    
    def test_train_validation(self):
        """ Perform a train/validation test"""
        
        # Prepare dataset
        train_loader, val_loader = utils.prepare_data_loaders()

        # Build the model
        net = nn.DataParallel(self.model.Net(train_loader.dataset.num_tokens)).cuda()
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
        tracker = utils.Tracker()
        config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        train_writer = SummaryWriter(config.visualization_dir + 'train')
        val_writer = SummaryWriter(config.visualization_dir + 'val')
        
        # Separate objects for train and validation
        vqa_trainer = TrainValidation(net, train_loader, optimizer, tracker, train_writer, train=True, prefix='train')
        vqa_validator = TrainValidation(net, val_loader, optimizer, tracker, val_writer, train=False, prefix='val')

        for epoch in range(config.num_epochs):
            _ = vqa_trainer.run_single_epoch()
            r = vqa_validator.run_single_epoch()

            results = {
                'name': 'model_training.pth',
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2],
                },
                'vocab': train_loader.dataset.vocab,
            }
            torch.save(results, "model_training.pth")

        train_writer.close()
        val_writer.close()



