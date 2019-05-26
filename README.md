## Dust
### Introduction
This is **Dust**. A **PyTorch** based framework for building visual question answering models. Dust provides high-level API to seamlessly integrate your VQA models with pre-defined data, image preprocessing and natural language proprocessing pipelines.

#### Features
*   Image preprocessing
*   Text preprocessing
*   Data Handling (MS-COCO Only)
*   Real-time Loss and Accuracy Tracker
*   VQA Evaluation 
*   Extendable Built-in Model Warehouse


#### Install


#### Quick Examples

* Image preprocessing with a custom CNN architecture:

```
import torch.nn as nn
from dust.backend.cnn.resnet import resnet
from dust.preprocessing.vision import Vision
from dust.tests import config


# define your own image feature extractor which inherits pytorch nn module
class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        self.model = resnet.resnet101(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

# perform image preprocessing with your custom CNN
my_cnn = myCNN().cuda()
visual_preprocessor = Vision(
    transforms_to_apply=['none'],
    cnn_to_use=myCNN,
    path_to_save=config.preprocessed_path,
    path_to_train_images=config.train_path,
    path_to_val_images=config.val_path,
    batch_size=config.preprocess_batch_size,
    image_size=config.image_size,
    keep_central_fraction=config.central_fraction,
    num_threads_to_use=8
)
visual_preprocessor.initiate_visual_preprocessing()

```
The above code will load your defined model (either pre-trained or from scratch) and saves the extracted features to **path_to_save**.

* Text preprocessing with one of the available models (GloVe or Word2Vec):

```
from dust.preprocessing.language import Language
from dust.tests import config

language_preprocessor = Language(
    max_answers=config.max_answers,
    save_vocab_to=config.vocabulary_path
)
language_preprocessor.initiate_vocab_extraction()

# Remove this line if you want to use Word2Vec embeddings
language_preprocessor.extract_glove_embeddings(
    dims=50,
    path_to_pretrained_embeddings=config.glove_embeddings,
    save_vectors_to=config.glove_processed_vectors,
    save_words_to=config.glove_words,
    save_ids_to=config.glove_ids
)
```
The above code will tokenize the VQA dataset and saves extracted GloVe vectors to **save_vectors_to**.
If you want to use random Word2Vec embeddings, simply remove the call to **extract_glove_embeddings**; the framework will automatically switch to randomly initialized Word2Vec embeddings.
### Contribution
Please feel free to contribute to the project. You may send a pull-request or drop me an email to talk more ([hexpheus@gmail.com](hexpheus@gmail.com))