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
|Adam `lr=0.001`| `[0.6022]` | `[1.8572]` | `[0.911]` | `[0.473]`|

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
|Baseline 2| ResNet-50 | Word2Vec (300) | FC `[8192, 10]` | `128 * 128` | `4 * 4 * 2048 ` | `25` | `512`|`[-]`|`[-]`|

</center>

#### Results

<center>

| Optimizer        | Train Loss | Validation Loss  | Train Accuracy | Validation Accuracy|
| ---------- |:---------:| :-----:| :------:| :-------: |
|Adam `lr=0.001`| `[0.5895]` | `[1.9466]` | `[0.917]` | `[0.445]`|


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="B2 Accuracy" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b2/accuracy.PNG">  |  <img width="1604" alt="B2 Loss" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b2/loss.PNG">
</center>


## VQA OE Baseline 3
In this baseline model, we will use pre-trained ResNet-101 weights. Note that the weights are extracted from the Caffe model of ResNet-101. A **word2vec** model with 300 embedding features are implemented to extract the question embeddings. Visual features are then concatenated with word embeddings and passed through a classifier with **8192** hidden nodes and classified to 10 answers.

#### Configuration Table

<center>

| Model        | Visual Features | Embeddings  | Classifier | Image Size | Visual FM Size | Epochs | Batch Size| Regularization | Attention |
| :----------: |:---------:| :-----:|:-------:|:------:|:-----:|:----:|:----:|:------:|:----:|
|Baseline 3| ResNet-101 | Word2Vec (300) | FC `[8192, 10]` | `128 * 128` | `4 * 4 * 2048 ` | `25` | `512`|`[-]`|`[-]`|

</center>

#### Results

<center>

##### Unofficial Evaluation

| Optimizer        | Train Loss | Validation Loss  | Train Accuracy | Validation Accuracy|
| ---------- |:---------:| :-----:| :------:| :-------: |
|Adam `lr=0.001`| `[0.594]` | `[1.930]` | `[0.915]` | `[0.456]`

##### Official Evaluation

Here is the evauluation results on the **val2014** split.

| Model        | All | Yes/No  | Number | Other |
| ---------- |:---------:| :-----:| :------:| :-------: |
|Baseline 3| `[0.431]` | `[0.676]` | `[0.286]` | `[0.283]`|


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="B3 Accuracy" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b3/accuracy.PNG">  |  <img width="1604" alt="B3 Loss" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b3/loss.PNG">
</center>

## VQA OE Baseline 4
In this baseline model, we will use pre-trained ResNet-101 weights. Note that the weights are extracted from the Caffe model of ResNet-101. A **GloVe** model with 50 embedding features are implemented to extract the question embeddings. Visual features are then concatenated with word embeddings and passed through a classifier with **8192** hidden nodes and classified to 10 answers.

#### Configuration Table

<center>

| Model        | Visual Features | Embeddings  | Classifier | Image Size | Visual FM Size | Epochs | Batch Size| Regularization | Attention |
| :----------: |:---------:| :-----:|:-------:|:------:|:-----:|:----:|:----:|:------:|:----:|
|Baseline 4| ResNet-101 | GloVe (50) | FC `[8192, 10]` | `128 * 128` | `4 * 4 * 2048 ` | `25` | `512`|`[-]`|`[-]`|

</center>

#### Results

<center>

##### Unofficial Evaluation

| Optimizer        | Train Loss | Validation Loss  | Train Accuracy | Validation Accuracy|
| ---------- |:---------:| :-----:| :------:| :-------: |
|Adam `lr=0.001`| `[0.736]` | `[2.006]` | `[0.865]` | `[0.434]`|


##### Official Evaluation

Here is the evauluation results on the **val2014** split.

| Model        | All | Yes/No  | Number | Other |
| ---------- |:---------:| :-----:| :------:| :-------: |
|Baseline 4| `[0.434]` | `[0.671]` | `[0.294]` | `[0.290]`|


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="B4 Accuracy" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b4/accuracy.png">  |  <img width="1604" alt="B4 Loss" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/results/b4/loss.png">
</center>