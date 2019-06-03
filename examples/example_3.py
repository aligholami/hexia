import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from dustorch.backend.utilities import utils
from dustorch.tests import config
from dustorch.backend.monitoring.tracker import Tracker
from dustorch.runtime.train_val import TrainValidation
from dustorch.vqa.models.joint import M_Resnet18_randw2v_NoAtt_Concat as model

# Prepare dataset
train_loader, val_loader = utils.prepare_data_loaders(path_to_feature_maps=config.preprocessed_path,
                                                      batch_size=config.batch_size, num_workers=config.data_workers)

# Build the model
net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
tracker = Tracker()
train_writer = SummaryWriter(config.visualization_dir + 'train')
val_writer = SummaryWriter(config.visualization_dir + 'val')

# Separate objects for train and validation (enables auto-resume on valid path settings)
vqa_trainer = TrainValidation(net, train_loader, optimizer, tracker, train_writer, train=True, prefix='train',
                              latest_vqa_results_path=config.latest_vqa_results_path)
vqa_validator = TrainValidation(net, val_loader, optimizer, tracker, val_writer, train=False, prefix='val',
                                latest_vqa_results_path=config.latest_vqa_results_path)

best_loss = 10.0
best_accuracy = 0.1
for epoch in range(config.num_epochs):
    _ = vqa_trainer.run_single_epoch()
    r = vqa_validator.run_single_epoch()

    if r['epoch_accuracy'] > best_accuracy and r['epoch_loss'] < best_loss:

        # Update best accuracy and loss
        best_accuracy = r['epoch_accuracy']
        best_loss = r['epoch_loss']

        # Clear path from previus saved models and pre-evaluation files
        try:
            os.remove(config.best_vqa_weights_path)
            os.remove(config.best_vqa_answers_to_eval)
        except FileNotFoundError as fe:
            pass

        # Save the new model weights
        weights = net.state_dict()
        torch.save(weights, config.best_vqa_weights_path)

        # Save answ, idxs and qids for later evaluation
        utils.save_for_vqa_evaluation(r['answ'], r['ids'], r['qids'])

    checkpoint = {
        'epoch': r['epoch'],
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tracker': tracker.to_dict(),
        'vocab': train_loader.dataset.vocab,
        'train_iters': r['train_iters'],
        'val_iters': r['val_iters'],
        'prefix': r['prefix'],
        'train': r['train'],
        'loader': r['loader']
    }

    torch.save(checkpoint, config.latest_vqa_results_path)

train_writer.close()
val_writer.close()
