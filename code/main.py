from vqa_san import VQA_SAN
from utils import clean_sentence
from data_generator import DataGenerator

# vqa_model = VQA_SAN()

# Clean sentence test
print(clean_sentence('sentence? afa . . puncha?'))

gen = DataGenerator(image_path='../data/train/images/full-image-dir', q_path='../data/train/questions/v2_OpenEnded_mscoco_train2014_questions.json', a_path='../data/train/answers/v2_mscoco_train2014_annotations.json', image_rescale=1, image_horizontal_flip=False, image_target_size=(150, 150))

mb_gen = gen.mini_batch_generator()


for i in range(20):
    print(mb_gen.__next__())


# data_gen = vqa_model.get_data()

# for i in range(10):
#     print(data_gen.__next__())
