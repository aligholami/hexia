from train_val import VQATrainValidation
import config

# Create the trainng and validation object
train_val = VQATrainValidation(config.learning_rate, num_epochs=config.num_epochs)
