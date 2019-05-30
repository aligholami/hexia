import unittest
import os
from dust.runtime.train_val import TrainValidation
from dust.preprocessing.vision import Vision
from dust.preprocessing.language import Language
from dust.backend.monitoring.tracker import Tracker
# from dust.vqa.evaluation.evaluator import VQAEvaluator
import torch
import torch.nn as nn
import torch.optim as optim
from dust.tests import config
from dust.backend.utilities import utils
from dust.vqa.models.joint import M_ResNet101_randw2v_NoAtt_LSTM as model
from tensorboardX import SummaryWriter
from dust.backend.cnn.resnet import resnet as caffe_resnet
from dust.vqa.models.joint import M_ResNet101_randw2v_NoAtt_LSTM as model

class VQAPITest(unittest.TestCase):

    # def test_visual_preprocessing(self):
    #     """
    #         Performs a visual preprocessing test.
    #     """
    #
    #     # Create a custom CNN class
    #     class ResNetCNN(nn.Module):
    #
    #         def __init__(self):
    #             super(ResNetCNN, self).__init__()
    #             self.model = caffe_resnet.resnet101(pretrained=True)
    #
    #             def save_output(module, input, output):
    #                 self.buffer = output
    #
    #             self.model.layer4.register_forward_hook(save_output)
    #
    #         def forward(self, x):
    #             self.model(x)
    #             return self.buffer
    #
    #     # Create an instance of custom CNN
    #     myCNN = ResNetCNN().cuda()
    #
    #     visual_preprocessor = Vision(
    #         transforms_to_apply=['none'],
    #         cnn_to_use=myCNN,
    #         path_to_save=config.preprocessed_path,
    #         path_to_train_images=config.train_path,
    #         path_to_val_images=config.val_path,
    #         batch_size=config.preprocess_batch_size,
    #         image_size=config.image_size,
    #         keep_central_fraction=config.central_fraction,
    #         num_threads_to_use=8
    #     )
    #
    #     visual_preprocessor.initiate_visual_preprocessing()

    # def test_language_preprocessing(self):
    #     """
    #         Performs a textual preprocessing test.
    #     """
    #
    #     language_preprocessor = Language(
    #         max_answers=config.max_answers,
    #         save_vocab_to=config.vocabulary_path
    #     )
    #
    #     language_preprocessor.initiate_vocab_extraction()
    #
    #     language_preprocessor.extract_glove_embeddings(
    #         dims=50,
    #         path_to_pretrained_embeddings=config.glove_embeddings,
    #         save_vectors_to=config.glove_processed_vectors,
    #         save_words_to=config.glove_words,
    #         save_ids_to=config.glove_ids
    #     )

    #
    def test_train_validation(self):
        """
            Perform a train/validation test.
        """

        # Prepare dataset
        train_loader, val_loader = utils.prepare_data_loaders(path_to_feature_maps=config.preprocessed_path, batch_size=config.batch_size, num_workers=config.data_workers)

        # Select the model
        # Build the model
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
        tracker = Tracker()
        # config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        train_writer = SummaryWriter(config.visualization_dir + 'train')
        val_writer = SummaryWriter(config.visualization_dir + 'val')

        # Separate objects for train and validation (enables auto-resume on valid path settings)
        vqa_trainer = TrainValidation(net, train_loader, optimizer, tracker, train_writer, train=True, prefix='train', latest_vqa_results_path=config.latest_vqa_results_path)
        vqa_validator = TrainValidation(net, val_loader, optimizer, tracker, val_writer, train=False, prefix='val', latest_vqa_results_path=None)

        best_loss = 10.0
        best_accuracy = 0.1
        epoch = 0

        while epoch < config.num_epochs:

            _ = vqa_trainer.run_single_epoch()
            r = vqa_validator.run_single_epoch()

            # Check if resumed
            resume_possible = r['resume_status']
            if resume_possible:
                # Continue epochs
                epoch = r['epoch']

            if r['epoch_accuracy'] > best_accuracy and r['epoch_loss'] < best_loss:

                # Update best accuracy and loss
                best_accuracy = r['epoch_accuracy']
                best_loss = r['epoch_loss']

                # Clear path from previus saved models and pre-evaluation files
                try:
                    os.remove(config.best_vqa_weights_path)
                    os.remove(config.best_vqa_answers_to_eval)
                except:
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
                'vocab': train_loader.dataset.vocab,
                'train_iters': r['train_iters'],
                'val_iters': r['val_iters'],
                'prefix': r['prefix'],
                'train': r['train'],
                'loader': r['loader']
            }

            torch.save(checkpoint, config.latest_vqa_results_path)

            # Update epochs
            epoch += 1

        train_writer.close()
        val_writer.close()

    # def test_vqa_evaluation(self):
    #     """
    #         Tests official VQA evaluation.
    #     """
    #
    #     v_evaluator = VQAEvaluator(
    #         data_directory=config.data_directory,
    #         best_model_results_directory=config.best_vqa_answers_to_eval
    #     )
    #
    #     v_evaluator.evaluate_best_vqa_model()

if __name__ == "__main__":
    unittest.main()

