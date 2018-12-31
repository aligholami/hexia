import cv2
from utils import get_image_id
import json
import os
from math import ceil

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
        one_hot_confidences = []
        for x in confidence_list:
            if x == 'yes':
                one_hot_confidences.append([1, 0, 0])
            elif x == 'maybe':
                one_hot_confidence.append([0, 1, 0])
            else:
                one_hot_confidence.append([0, 0, 1])
                
        return one_hot_confidences

    def mini_batch_generator(self):
        
        # Generate a batch of images
        # for each image in the batch generate 

        # For each file in the image list
        for image_name in self.image_list:   

            # Read image from directory
            img = cv2.imread(os.path.join(self.image_path, image_name))

            # Extract the image ID
            img_id = get_image_id(image_name)

            # Extract questions, answers, and labels (confidences) from the JSON files
            img_question_list = []
            img_answer_list = []
            iqa_label_list = []

            for question in self.q_data['questions']:
                if(question['image_id'] == int(img_id)):
                    for annotation in self.a_data['annotations']:
                        if(annotation['question_id'] == question['question_id']):
                            answer_no = 0
                            for answer in annotation['answers']:
                                batch_item = {}
                                batch_item['image'] = img
                                batch_item['question'] = question['question']
                                batch_item['answer'] = annotation['answers'][answer_no]['answer']
                                batch_item['iqa_label'] = annotation['answers'][answer_no]['answer_confidence']
                                answer_no = answer_no + 1
                                yield (batch_item['image'], batch_item['question'], batch_item['answer'], batch_item['iqa_label'])
                    
            # Grab one-hot vectors for the confidences list
            # one_hot_iqa = confidence_to_one_hot(iqa_label_list)
