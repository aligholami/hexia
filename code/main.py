from vqa_san import VQA_SAN

vqa_model = VQA_SAN()


data_batch = vqa_model.get_data().__next__()

print(data_batch)


