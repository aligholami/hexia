# Visual Question Answering based on Stacked Attention Networks
The goal of this repository is to implement and compare different novel architectures of Visual Question Answering. Initially, several strong baselines are implemented for the **OpenEnded** visual question answering task based on the **MS COCO 2014** dataset. The implementation is mainly in **PyTorch**. 

## VQA OE Baseline 1
In this baseline model, we will use pre-trained ResNet-18 weights. Note that the weights are extracted from the Caffe model of ResNet-18. A **word2vec** model with 300 embedding features are implemented to extract the question embeddings. Visual features are then concatenated with word embeddings and passed through a classifier with **8192** hidden nodes and classified to 10 answers.

#### Configuration Table

#### Results
