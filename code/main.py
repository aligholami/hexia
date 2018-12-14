from vqa_san import VQA_SAN

vqa_model = VQA_SAN()


image_ids = vqa_model.get_data().__next__()

print(image_ids)

