from vqa_san import VQA_SAN

# Instantiate the VQA class
vqa_model = VQA_SAN()

# Build computation graph
vqa_model.build_model()

# Start training
vqa_model.train_and_validate(num_epochs=20)