import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import h5py
from tqdm import tqdm
from hexia.tests import config
from hexia.backend.dataset import data
from hexia.backend.utilities import utils

class Vision:

    cudnn.benchmark = True

    def __init__(self, transforms_to_apply, cnn_to_use, path_to_save, path_to_train_images, path_to_val_images, batch_size, image_size, keep_central_fraction, num_threads_to_use):
        self.transforms_to_apply = transforms_to_apply
        self.cnn_to_use = cnn_to_use
        self.path_to_save = path_to_save
        self.batch_size = batch_size
        self.image_size = image_size
        self.keep_central_fraction = keep_central_fraction
        self.num_threads_to_use = num_threads_to_use
        self.path_to_train_images = path_to_train_images
        self.path_to_val_images = path_to_val_images

    def create_data_loader(self, *paths):
        """ Create a united PyTorch COCO data loader for every given path in the arguments"""
        transform = utils.get_transform(self.image_size, self.keep_central_fraction)
        datasets = [data.CocoImages(path, transform=transform) for path in paths]
        dataset = data.Composite(*datasets)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_threads_to_use,
            shuffle=False,
            pin_memory=True,
        )

        features_shape = (
            len(data_loader.dataset),
            config.output_features,
            config.output_size,
            config.output_size
        )

        return data_loader, features_shape

    def initiate_visual_preprocessing(self):
        """ Extract feature maps and save them to drive (to path_to_save)"""

        net = self.cnn_to_use
        net.eval()

        loader, features_shape = self.create_data_loader(self.path_to_train_images, self.path_to_val_images)

        with h5py.File(self.path_to_save, libver='latest') as fd:
            features = fd.create_dataset('features', shape=features_shape, dtype='float16')
            coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

            i = 0
            for ids, imgs in tqdm(loader):
                imgs = Variable(imgs.cuda(async=True), volatile=True)
                out = net(imgs)

                j = i + imgs.size(0)
                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j
