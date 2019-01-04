<<<<<<< HEAD
import numpy as np
from collections import defaultdict
import re

def get_image_id(filename):

    id = filename.rsplit('.jpg')[0]
    id = id.split('_')[2]
    id = id.lstrip('0')
    return id
        

def clean_sentence(sentence):
    sentence = sentence.replace("the", "")
    sentence = sentence.replace("and", "")
    sentence = sentence.replace("/", " ")
    return re.sub(r'[^\w\s]', '', sentence)

def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """

    print("Loading embedding from disks...")

    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    
    with open(glove_filename, 'r', encoding='utf-8') as glove_file:
        for (i, line) in enumerate(glove_file):
            
            split = line.split(' ')
            
            word = split[0]
            
            # Save words for further usages
            # with open('vocab_only.txt', 'a', encoding="utf-8") as vocab_file:
            #     vocab_file.write(word+'\n')

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            
            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        print("Embedding loaded from disks.")
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
=======
import numpy as np
from collections import defaultdict
import re

def get_image_id(filename):

    id = filename.rsplit('.jpg')[0]
    id = id.split('_')[2]
    id = id.lstrip('0')
    return id
        

def clean_sentence(sentence):
    sentence = sentence.replace("the", "")
    sentence = sentence.replace("and", "")
    sentence = sentence.replace("/", " ")
    return re.sub(r'[^\w\s]', '', sentence)

def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """

    print("Loading embedding from disks...")

    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    
    with open(glove_filename, 'r', encoding='utf-8') as glove_file:
        for (i, line) in enumerate(glove_file):
            
            split = line.split(' ')
            
            word = split[0]
            
            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            
            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        print("Embedding loaded from disks.")
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
>>>>>>> ad30bb6cf245299f788a8b049bc1857d82952da5
        return word_to_embedding_dict