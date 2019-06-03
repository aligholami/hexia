import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dustorch.tests import config
from tqdm import tqdm
from dustorch.backend.utilities import utils
from dustorch.backend.dataset.data import DataLoadUtils
from tensorboardX import SummaryWriter
import warnings
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainValidation:

    def __init__(self, model, loader, optimizer, tracker, writer, train, prefix, latest_vqa_results_path):
        """ Initialize the train and validation over the given model and optimizer """

        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.tracker = tracker
        self.train = train
        self.writer = writer
        self.prefix = prefix
        self.epochs_passed = 0
        self.train_iterations = 0
        self.val_iterations = 0
        self.latest_vqa_results_path = latest_vqa_results_path
        self.resume_possible = True

    def run_single_epoch(self, epoch):
        """ Run the given model settings for one epoch """

        # register number of passed epochs
        self.epochs_passed = epoch

        # check if we are eligible to perform a training resume
        if self.resume_possible:
            self.auto_resume()

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
            acc = utils.batch_accuracy(out.data, a.data).cpu()

            if self.train:
                utils.update_learning_rate(self.optimizer, self.train_iterations)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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

            # return answers, idxs, question ids, epoch and epoch accuracy and epoch loss
            epoch_results = {
                'epoch': self.epochs_passed,
                'answ': answ,
                'ids': idxs,
                'qids': qids,
                'epoch_accuracy': acc_tracker.mean.value.item(),
                'epoch_loss': loss_tracker.mean.value,
                'train_iters': self.train_iterations,
                'val_iters': self.val_iterations,
                'prefix': self.prefix,
                'train': self.train,
                'resume_status': self.resume_possible
            }

        else:
            # No need to return answers in training mode, just training information
            epoch_results = {
                'epoch': self.epochs_passed,
                'epoch_accuracy': acc_tracker.mean.value.item(),
                'epoch_loss': loss_tracker.mean.value,
                'train_iters': self.train_iterations,
                'val_iters': self.val_iterations,
                'prefix': self.prefix,
                'train': self.train,
                'resume_status': self.resume_possible
            }

        return epoch_results

    def resume_from_checkpoint(self, checkpoint):
        try:
            self.epochs_passed = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_iterations = checkpoint['train_iters']
            self.val_iterations = checkpoint['val_iters']
            # Re-define a writer to continue writing the train/validation/other instances status
            self.writer = SummaryWriter(config.visualization_dir + self.prefix)
            # Disable file resume for the current instance
            self.resume_possible = False

            print("Loaded model status: ")
            print("=====================")
            print("Resuming from epoch {}".format(self.epochs_passed))
            print("Resuming {}".format(self.prefix))
            print("Training status is {}".format(self.train))
            print("{} resumed.".format(self.prefix))
            print("=====================")

        except KeyError as keye:
            print("Something went wrong in checkpoint file keys: {0}".format(keye.value))

    def auto_resume(self):

        if not self.latest_vqa_results_path:
            # Disable file resume if path is not specified (None)
            self.resume_possible = False
        else:
            if os.path.exists(self.latest_vqa_results_path):
                print("Looking for resume file at {}".format(self.latest_vqa_results_path))
                try:
                    checkpoint = torch.load(self.latest_vqa_results_path)
                    self.resume_from_checkpoint(checkpoint)
                except BaseException as bex:
                    print("Invalid checkpoint at {}".format(self.latest_vqa_results_path))
            else:
                # Disable file resume for the current instance
                self.resume_possible = False

