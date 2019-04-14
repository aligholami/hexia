import unittest
# from vqapi.runtime.train_val import TrainValidation
from vqapi.preprocessing.vision import Vision
import torch
import torch.nn as nn
import torch.optim as optim
from vqapi.tests import config
from vqapi.backend.utilities import utils
# from tensorboardX import SummaryWriter
from vqapi.backend.cnn.resnet import resnet as caffe_resnet
# from models import M_ResNet101_randw2v_NoAtt_LSTM as model

class VQAPITest(unittest.TestCase):

    # def test_train_validation(self):
    #     """ Perform a train/validation test"""
        
    #     # Prepare dataset
    #     train_loader, val_loader = utils.prepare_data_loaders()

    #     # Build the model
    #     net = nn.DataParallel(self.model.Net(train_loader.dataset.num_tokens)).cuda()
    #     optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    #     tracker = utils.Tracker()
    #     config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    #     train_writer = SummaryWriter(config.visualization_dir + 'train')
    #     val_writer = SummaryWriter(config.visualization_dir + 'val')
        
    #     # Separate objects for train and validation
    #     vqa_trainer = TrainValidation(net, train_loader, optimizer, tracker, train_writer, train=True, prefix='train')
    #     vqa_validator = TrainValidation(net, val_loader, optimizer, tracker, val_writer, train=False, prefix='val')

    #     for epoch in range(config.num_epochs):
    #         _ = vqa_trainer.run_single_epoch()
    #         r = vqa_validator.run_single_epoch()

    #         results = {
    #             'name': 'model_training.pth',
    #             'tracker': tracker.to_dict(),
    #             'config': config_as_dict,
    #             'weights': net.state_dict(),
    #             'eval': {
    #                 'answers': r[0],
    #                 'accuracies': r[1],
    #                 'idx': r[2],
    #             },
    #             'vocab': train_loader.dataset.vocab,
    #         }
    #         torch.save(results, "model_training.pth")

    #     train_writer.close()
    #     val_writer.close()

    def test_visual_preprocessing(self):
        """
            Performs a visual preprocessing test.
        """

        # Create a custom CNN class
        class ResNetCNN(nn.Module):

            def __init__(self):
                super(ResNetCNN, self).__init__()
                self.model = caffe_resnet.resnet101(pretrained=True)
                
                def save_output(module, input, output):
                    self.buffer = output

                self.model.layer4.register_forward_hook(save_output)

            def forward(self, x):
                self.model(x)
                return self.buffer

        # Create an instance of custom CNN
        myCNN = ResNetCNN().cuda()

        visual_preprocessor = Vision(
            transforms_to_apply=['none'],
            cnn_to_use=myCNN,
            path_to_save=config.preprocessed_path,
            path_to_train_images=config.train_path,
            path_to_val_images=config.val_path,
            batch_size=config.preprocess_batch_size,
            image_size=config.image_size,
            keep_central_fraction=config.central_fraction,
            num_threads_to_use=8
        )

        visual_preprocessor.initiate_visual_preprocessing()
    
    def test_language_preprocessing(self):
        """
            Performs a textual preprocessing test.
        """
        

if __name__ == "__main__":
    unittest.main()

