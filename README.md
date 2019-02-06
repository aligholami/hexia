# Visual Question Answering based on Stacked Attention Networks
The goal of this repository is to implement and compare different novel architectures of Visual Question Answering. Initially, several strong baselines are implemented for the **OpenEnded** visual question answering task based on the **MS COCO 2014** dataset. The implementation is mainly in **PyTorch**. 

## VQA OE Baseline 1
In this baseline model, we will use pre-trained ResNet-18 weights. Note that the weights are extracted from the Caffe model of ResNet-18. A **word2vec** model with 300 embedding features are implemented to extract the question embeddings. Visual features are then concatenated with word embeddings and passed through a classifier with **8192** hidden nodes and classified to 10 answers.

#### Configuration Table

<center>

| Model        | Visual Features | Embeddings  | Classifier | Image Size | Visual FM Size | Epochs | Batch Size| Regularization | Attention|
| :----------: |:---------:| :-----:|:-------:|:------:|:-----:|:----:|:----:|:-----:|:-----:|
|Baseline 1| ResNet-18 | Word2Vec (300) | FC `[8192, 10]` | `128 * 128` | `4 * 4 * 512 ` | `25` | `512`|`[-]`|`[-]`|

</center>

#### Results

<center>

| Optimizer        | Train Loss | Validation Loss  | Train Accuracy | Validation Accuracy|
|---------- |:---------:| :-----:| :------:| :-------: |
|Adam `lr=0.001`| `[0.6022]` | `[1.8572]` | `[0.91]` | `[0.47]`|

</center>

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="B1 Accuracy" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b1/accuracy.PNG">  |  <img width="1604" alt="B1 Loss" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b1/loss.PNG">

## VQA OE Baseline 2
In this baseline model, we will use pre-trained ResNet-50 weights. Note that the weights are extracted from the Caffe model of ResNet-50. A **word2vec** model with 300 embedding features are implemented to extract the question embeddings. Visual features are then concatenated with word embeddings and passed through a classifier with **8192** hidden nodes and classified to 10 answers.

#### Configuration Table

<center>

| Model        | Visual Features | Embeddings  | Classifier | Image Size | Visual FM Size | Epochs | Batch Size| Regularization | Attention |
| :----------: |:---------:| :-----:|:-------:|:------:|:-----:|:----:|:----:|:------:|:----:|
|Baseline 2| ResNet-50 | Word2Vec (300) | FC `[8192, 10]` | `128 * 128` | `4 * 4 * 512 ` | `25` | `512`|`[-]`|`[-]`|

</center>

#### Results

<center>

| Optimizer        | Train Loss | Validation Loss  | Train Accuracy | Validation Accuracy|
| ---------- |:---------:| :-----:| :------:| :-------: |
|Adam `lr=0.001`| `[0.6022]` | `[RUNNING]` | `[RUNNING]` | `[RUNNING]`|

</center>