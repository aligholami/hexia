import cv2
from utils import get_image_id
import json
import os

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
        
        # Load all image files of directory into the memory
        self.image_list = os.listdir(self.image_path)

        with open(self.q_path, encoding='utf-8') as q_file:
            self.q_data = json.loads(q_file.read())

        with open(self.a_path, encoding='utf-8') as a_file:
            self.a_data = json.loads(a_file.read())
    
    def confidence_to_one_hot(self, confidence_list):
        
        # confidences for (yes) / (maybe, no)
        one_hot_confidences = [0, 0]

        one_hot_confidences = [if x is 'yes' [1, 0] else [0, 1] for x in confidence_list]

        return one_hot_confidences

    def mini_batch_generator(self, batch_size):
        
        # Generate a batch of images
        # for each image in the batch generate 

        # For each file in the image list
        for image_name in self.image_list:   

            # Read image from directory
            img = cv2.imread(os.path.join(self.image_path, image_name))

            # Extract the image ID
            img_id = get_image_id(image_name)

            # Extract questions and answers from the JSON files
            img_question_list = []
            img_answer_list = []
            iqa_label_list = []

            for entry in self.q_data['questions']:
                if(entry['image_id'] == int(img_id)):
                    img_question_list.append(entry['question'])
            
            for entry in self.a_data['annotations']:
                if(entry['image_id'] == int(img_id)):
                    for i in entry['answers']:
                        img_answer_list.append(entry['answers'][i]['answer'])
                        iqa_label_list.append(entry['answers'][i]['confidence'])
            
            # Grab one-hot vectors for the confidences list
            one_hot_iqa = confidence_to_one_hot(iqa_label_list)

            for item in range(len(iqa_label_list)):
                batch_item = {}
                batch_item['image'] = img
                batch_item['question'] = img_question_list[item]
                batch_item['answer'] = img_answer_list[item]
                batch_item['iqa_label'] = one_hot_iqa[item]     # one-hot labels
                yield (batch_item['image'], batch_item['question'], batch_item['answer'], batch_item['iqa_label'])