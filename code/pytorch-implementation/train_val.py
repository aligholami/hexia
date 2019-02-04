import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import model
import config
import data
import utils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_iterations = 0


class VQATrainValidation:
    __skip_steps = 100

    def __init__(self, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def __prepare_data_loaders__(self):

        train_loader = data.get_loader(train=True)
        val_loader = data.get_loader(val=True)

        return train_loader, val_loader

    @staticmethod
    def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
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

        tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
        acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

        log_softmax = nn.LogSoftmax().cuda()
        for v, q, a, idx, q_len in tq:
            var_params = {
                'volatile': not train,
                'requires_grad': False,
            }
            v = Variable(v.cuda(async=True), **var_params)
            q = Variable(q.cuda(async=True), **var_params)
            a = Variable(a.cuda(async=True), **var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)

            out = net(v, q, q_len)
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()

            if train:
                global total_iterations
                VQATrainValidation.update_learning_rate(optimizer, total_iterations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_iterations += 1
            else:
                # store information about evaluation of this minibatch
                _, answer = out.data.cpu().max(dim=1)
                answ.append(answer.view(-1))
                accs.append(acc.view(-1))
                idxs.append(idx.view(-1).clone())

            loss_tracker.append(loss.data.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        if not train:
            answ = list(torch.cat(answ, dim=0))
            accs = list(torch.cat(accs, dim=0))
            idxs = list(torch.cat(idxs, dim=0))
            return answ, accs, idxs

    def start(self):
        """
        Start training on the vqa model.
        :return: Statistical measures and reports.
        """

        # Prepare dataset
        train_loader, val_loader = self.__prepare_data_loaders__()

        # Build the model
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
        tracker = utils.Tracker()
        config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        for i in range(self.num_epochs):
            _ = VQATrainValidation.run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
            r = VQATrainValidation.run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

            results = {
                'name': 'target_name.pth',
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
            torch.save(results, "target_name.pth")
