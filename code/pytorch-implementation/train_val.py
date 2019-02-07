import torch
import torch.nn as nn
import torch.optim as optim
import config
import utils as utils
from tensorboardX import SummaryWriter
from models import M_Resnet101_Glove_NoAtt_Concat as model
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_validate():
    """
    Start training on the vqa model.
    :return: Statistical measures and reports.
    """

    # Prepare dataset
    train_loader, val_loader = utils.prepare_data_loaders()

    # Build the model
    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    train_writer = SummaryWriter(config.visualization_dir + 'train')
    val_writer = SummaryWriter(config.visualization_dir + 'val')

    for i in range(config.num_epochs):
        _ = utils.run(net, train_loader, optimizer, tracker, train_writer, train=True, prefix='train', epoch=i)
        r = utils.run(net, val_loader, optimizer, tracker, val_writer, train=False, prefix='val', epoch=i)

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


if __name__ == "__main__":
    train_and_validate()
