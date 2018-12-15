from image_generator import ImageGenerator
from utils import get_image_id
import json

# from text_generator import TextGenerator

class DataGenerator:

    def __init__(self, image_path, q_path, a_path, image_rescale, image_horizontal_flip, image_target_size):
        self.image_path = image_path
        self.q_path = q_path
        self.a_path = a_path
        self.image_rescale = image_rescale
        self.image_horizontal_flip = image_horizontal_flip
        self.image_target_size = image_target_size
        
        # Load Questions and Answers JSON into memory
        self.load_qa_into_mem()
    
    def load_qa_into_mem(self):

        with open(self.q_path, encoding='utf-8') as q_file:
            self.q_data = json.loads(q_file.read())

        with open(self.a_path, encoding='utf-8') as a_file:
            self.a_data = json.loads(a_file.read())
        
    def mini_batch_generator(self, batch_size):
        
        # Generate a batch of images
        # for each image in the batch generate 

        train_image_generator = ImageGenerator(path_to_generate=self.image_path, rescale=1, horizontal_flip=False, target_size=(150, 150))
        train_image_generator = train_image_generator.image_mb_generator(batch_size=batch_size)
        
        for image_batch in train_image_generator:
            batch_data = []
            idx = (train_image_generator.batch_index - 1) * train_image_generator.batch_size
            target_files = train_image_generator.filenames[idx: idx + train_image_generator.batch_size]
            image_batch_ids = get_image_id(target_files)
            img_no = 0

            for img_id in image_batch_ids:
                img_question_list = []
                img_answer_list = []

                for entry in self.q_data['questions']:
                    if(entry['image_id'] == int(img_id)):
                        img_question_list.append(entry['question'])
                
                for entry in self.a_data['annotations']:
                    if(entry['image_id'] == int(img_id)):
                        img_answer_list.append(entry['answers'][0]['answer'])

                for item in range(len(img_question_list)):
                    batch_item = {}
                    batch_item['question'] = img_question_list[item]
                    batch_item['answer'] = img_answer_list[item]
                    batch_item['image'] = image_batch[img_no]
                    batch_data.append(batch_item)
            
                img_no += 1
            
            yield batch_data