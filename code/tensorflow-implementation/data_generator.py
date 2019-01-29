import cv2
import numpy as np
from utils import get_file_list_in_dir, get_image_id, get_image_name_in_dir, clean_sentence, confidence_to_one_hot
import json as json
from tqdm import tqdm
import pickle
import os


# from text_generator import TextGenerator

class DataGenerator:

    TRAIN_INIT_CODE = 2
    VAL_INIT_CODE = 3

    def __init__(self, image_path, q_path, a_path, p_path, image_rescale, image_horizontal_flip, image_target_size,
                 use_num_answers, init_code):
        self.data_items = []
        self.image_path = image_path
        self.q_path = q_path
        self.a_path = a_path
        self.image_rescale = image_rescale
        self.image_horizontal_flip = image_horizontal_flip
        self.image_target_size = image_target_size
        self.p_path = p_path
        self.use_num_answers = use_num_answers
        self.init_code = init_code

        # Load Questions and Answers JSON into memory
        self.load_qa_into_mem()
        self.prepare_generator_iterable()

    def get_data_items(self):
        return self.data_items

    def load_qa_into_mem(self):
        """
        Load json files to memory for further usage.
        """
        # Load all image files of directory into the memory
        self.image_list = get_file_list_in_dir(self.image_path)

        # Try loading questions/answers jsons (Compatible with any Python version)
        try:
            with open(self.q_path, encoding='utf-8') as q_file:
                self.q_data = json.loads(q_file.read())

            with open(self.a_path, encoding='utf-8') as a_file:
                self.a_data = json.loads(a_file.read())

        except Exception as version_exception:
            try:
                with open(self.q_path, "r") as q_file:
                    self.q_data = json.loads(q_file.read().decode("latin1").encode("utf8"))

                with open(self.a_path, "r") as a_file:
                    self.a_data = json.loads(a_file.read().decode("latin1").encode("utf8"))

            except Exception as loading_exception:
                pass

    def get_num_of_samples(self):
        """
        Return the number of training samples
        """
        num_samples = 0

        num_all_questions = len(self.q_data['questions'])
        num_answers_for_each_question = self.use_num_answers
        # num_images = len(os.listdir(self.image_path))

        num_samples = num_all_questions * num_answers_for_each_question

        return num_samples

    def mini_batch_generator_v1(self):
        """
        Generator for feeding data through Tensorflow dataset API.
        """
        # For each file in the image list
        for image_name in self.image_list:

            # Read image from directory
            img = cv2.imread(os.path.join(self.image_path, image_name))
            img = cv2.resize(img, (64, 64))

            # Normalize
            img = img / 255.0

            # Extract the image ID
            img_id = get_image_id(image_name)

            # Extract questions, answers, and labels (confidences) from the JSON files
            for question in self.q_data['questions']:
                if (question['image_id'] == int(img_id)):
                    for annotation in self.a_data['annotations']:
                        if (annotation['question_id'] == question['question_id']):

                            # Select first 3 answers only
                            for answer_no in range(self.use_num_answers):
                                batch_item = {}
                                batch_item['image'] = img
                                batch_item['sentence'] = clean_sentence(
                                    question['question'] + ' ' + annotation['answers'][answer_no]['answer'])
                                # batch_item['question'] = clean_sentence(question['question'])
                                # batch_item['answer'] = clean_sentence(annotation['answers'][answer_no]['answer'])
                                batch_item['iqa_label'] = confidence_to_one_hot(
                                    annotation['answers'][answer_no]['answer_confidence'])
                                answer_no = answer_no + 1
                                # print(len(batch_item['sentence']))
                                # print(len(batch_item['sentence'].split()))
                                yield np.array(batch_item['image'].flatten()), batch_item['sentence'], np.array(
                                    batch_item['iqa_label'])

    def mini_batch_generator_v2(self):
        """
        Generator for feeding data through Tensorflow dataset API.
        """

        # print("Inside the generator..")

        for data_item in self.data_items:

            for image_name, sentence, confidence in data_item:

                # Please remove this in future releases
                if(self.init_code == self.VAL_INIT_CODE):
                    image_name = image_name.replace("train", "val")
                    
                # print("Sequence: {}, {}, {}".format(image_name, sentence, confidence))

                # Read corresponding image from directory
                img = cv2.imread(os.path.join(self.image_path, image_name))
                img = cv2.resize(img, (64, 64))

                # Normalize
                img = img / 255.0

                img = np.array(img.flatten())

                yield img, sentence, confidence

    def get_data_list(self):

        return self.data_items

    def prepare_generator_iterable(self):
        """
        Prepares a list of tuples to use inside the generator
        """

        print("Loading Data Items: ")

        skip_steps = 200000
        pickle_file_addr = self.p_path

        # Check if pickle file exists
        if os.path.isfile(pickle_file_addr):

            with open(pickle_file_addr, 'rb') as f:

                while True:
                    try:
                        self.data_items.append(pickle.load(f))

                    except EOFError:
                        break

                    # print("Data Items on Disk.")

        else:
            for i, question in tqdm(enumerate(self.q_data['questions'])):
                image_id = question['image_id']
                image_name = get_image_name_in_dir(image_id, self.init_code)

                for annotation in self.a_data['annotations']:
                    if annotation['question_id'] == question['question_id']:
                        for answer_no in range(self.use_num_answers):
                            item = {'image_name': image_name, 'sentence': clean_sentence(
                                question['question'] + ' ' + annotation['answers'][answer_no]['answer']),
                                    'confidence': confidence_to_one_hot(
                                        annotation['answers'][answer_no]['answer_confidence'])}

                            data_item = (item['image_name'], item['sentence'], np.array(item['confidence']))
                            self.data_items.append(data_item)

                if (i % skip_steps) == 0:
                    # Save pickle
                    try:
                        with open(pickle_file_addr, 'ab') as f:
                            pickle.dump(self.data_items, f, protocol=2)

                            # Free some memory
                            self.data_items = []
                    except Exception as _:
                        pass

            # Write whats left behind
            try:
                with open(pickle_file_addr, 'ab') as f:
                    pickle.dump(self.data_items, f, protocol=2)
            except Exception as _:
                pass

        print("Loaded Data Items.")
