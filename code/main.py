from vqa_san import VQA_SAN

vqa_model = VQA_SAN()


data_gen = vqa_model.get_data()

for i in range(10):
    print(data_gen.__next__())
