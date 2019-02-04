import torch
import torch.nn as nn
import torch.optim as optim
import model
import config
import utils

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
    for i in range(config.num_epochs):
        _ = utils.run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = utils.run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

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


if __name__ == "__main__":
    train_and_validate()
