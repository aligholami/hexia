import torch.nn as nn
from dustorch.backend.cnn.resnet import resnet
from dustorch.preprocessing.vision import Vision
from dustorch.tests import config


# define your own image feature extractor which inherits pytorch nn module
class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        self.model = resnet.resnet101(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

# perform image preprocessing with your custom CNN
my_cnn = myCNN().cuda()
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

