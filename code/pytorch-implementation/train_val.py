import torch
import torch.nn as nn
from vqa import VQA
import data_loader as data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VQATrainValidation:
    __skip_steps = 100

    def __init__(self, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def __prepare_data_loaders__(self):

        train_loader = data.get_loader(train=True)
        val_loader = data.get_loader(val=True)

        return train_loader, val_loader

    def __build_model__(self):
        """
        Build the VQA model (Connections)
        :return: Model, Loss Function and Optimizer
        """

        # Define the VQA model
        vqa_model = VQA().to(device)

        # Define target loss
        loss_function = nn.CrossEntropyLoss()

        # Define the optimization method
        optimizer = torch.optim.Adam(vqa_model.parameters(), lr=self.learning_rate)

        return vqa_model, loss_function, optimizer

    def start(self):
        """
        Start training on the vqa model.
        :return: Statistical measures and reports.
        """

        # Prepare dataset
        train_loader, val_lodaer = self.__prepare_data_loaders__()

        # Build the model
        model, loss, opt = self.__build_model__()

        # Get number of training steps
        train_steps = len(train_loader)

        # Start training
        for epoch in range(self.num_epochs):
            for i, (image, sentence, label) in enumerate(train_loader):

                # Copy batch to GDDR
                image.to(device)
                sentence.to(device)
                label.to(device)

                # Forward pass
                prediction = model((image, sentence, label))
                loss_value = loss(prediction, label)

                # Optimize
                opt.zero_grad()
                loss_value.backward()
                opt.step()

                # Visualize training
                if (i + 1) % self.__skip_steps == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, train_steps, loss_value.item()))
