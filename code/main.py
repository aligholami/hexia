<<<<<<< HEAD
from vqa_san import VQA_SAN


# Instantiate the VQA class
vqa_model = VQA_SAN()

# Build computation graph
vqa_model.build_model()

# Start training
vqa_model.train_and_validate(batch_size=10, num_epochs=5)
=======
from vqa_san import VQA_SAN


# Instantiate the VQA class
vqa_model = VQA_SAN()

# Build computation graph
vqa_model.build_model()

# Start training
vqa_model.train_and_validate(batch_size=10, num_epochs=5)
>>>>>>> ad30bb6cf245299f788a8b049bc1857d82952da5
