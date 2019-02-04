import train_val
import config

# Create the trainng and validation object
train_val = train_val.VQATrainValidation(config.initial_lr, num_epochs=config.epochs)
train_val.start()
