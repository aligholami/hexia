import torch
import torch.nn as nn
import torch.optim as optim
import config
from vqapi.backend.utilities import utils
from tensorboardX import SummaryWriter
from models import M_ResNet101_randw2v_NoAtt_LSTM as model
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainValidation:

    def __init__(self, model, loader, optimizer, tracker, writer, train, prefix):
        """ Initialize the train and validation over the given model and optimizer """

        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.tracker = tracker
        self.train = train
        self.prefix = prefix
        self.epochs_passed = 0
        self.train_iterations = 0
        self.val_iterations = 0

    def run_single_epoch(self):
        """ Run the given model settings for one epoch """

        if self.train:
            self.model.train()
            tracker_class, tracker_params = self.tracker.MovingMeanMonitor, {'momentum': 0.99}
        else:
            self.model.eval()
            tracker_class, tracker_params = self.tracker.MeanMonitor, {}
            answ = []
            idxs = []
            accs = []
            qids = []

        tq = tqdm(self.loader, desc='{} E{:03d}'.format(self.prefix, self.epochs_passed), ncols=0)
        loss_tracker = self.tracker.track('{}_loss'.format(self.prefix), tracker_class(**tracker_params))
        acc_tracker = self.tracker.track('{}_acc'.format(self.prefix), tracker_class(**tracker_params))

        log_softmax = nn.LogSoftmax().cuda()
        for qid, v, q, a, idx, q_lens in tq:
            var_params = {
                'volatile': not self.train,
                'requires_grad': False,
            }
            v = Variable(v.cuda(async=True), **var_params)
            q = Variable(q.cuda(async=True), **var_params)
            a = Variable(a.cuda(async=True), **var_params)

            # used for sequence padding and packing
            q_lens = Variable(q_lens.cuda(async=True), **var_params)

            out = self.model(v, q, q_lens)
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = batch_accuracy(out.data, a.data).cpu()

            if self.train:
                update_learning_rate(self.optimizer, self.train_iterations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.train_iterations += 1

                # Write loss and accuracy to TensorboardX
                self.writer.add_scalar('/loss', loss.data.item(), self.train_iterations)
                self.writer.add_scalar('/accuracy', acc.mean(), self.train_iterations)

            else:
                # store information about evaluation of this minibatch
                _, answer = out.data.cpu().max(dim=1)
                qids.append(qid.view(-1).clone())
                answ.append(answer.view(-1))
                accs.append(acc.view(-1))
                idxs.append(idx.view(-1).clone())
                self.val_iterations += 1

                # Write loss and accuracy to TensorboardX
                self.writer.add_scalar('/loss', loss.data.item(), self.val_iterations)
                self.writer.add_scalar('/accuracy', acc.mean(), self.val_iterations)

            loss_tracker.append(loss.data.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        if not self.train:
            answ = list(torch.cat(answ, dim=0))
            accs = list(torch.cat(accs, dim=0))
            idxs = list(torch.cat(idxs, dim=0))
            qids = list(torch.cat(qids, dim=0))

            # Save results for vqa evaluation
            utils.save_for_vqa_evaluation(answ, idxs, qids, self.epochs_passed)

            return answ, accs, idxs