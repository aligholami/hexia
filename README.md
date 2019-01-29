# Visual Question Answering with Stacked Attention Networks

<div align="center">

<a href="https://www.python.org/downloads/release/python-360/"> <img src="https://img.shields.io/badge/python-3.6-blue.svg" alt="Python 3.6"/> </a>
<a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks&amp;utm_campaign=Badge_Grade"> <img src="https://api.codacy.com/project/badge/Grade/62aaec49f9294a46a74c65dacf599a37" alt="Codacy Badge"/> </a>
<a href="https://www.codefactor.io/repository/github/aligholami/visual-question-answering-with-stacked-attention-networks"><img src="https://www.codefactor.io/repository/github/aligholami/visual-question-answering-with-stacked-attention-networks/badge" alt="CodeFactor" /></a>

</div>

In this project, we will analyze different methods for building a VQA system. Our goal is to first produce promising results with basic archtiectures and then develop the model to a degree in which **reasoning** is being done at its best possible state.

## Baseline 1

## Configuration
Initially, the VQA model is trained and validated with the following configurations:

| Visual Features | Text Features | Merged Features | Normalization | Training Method | Cls. Method | Epochs | 
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| Pre-trained ResNet-50 | GloVe 50D | Concatenation | None | Adam (LR = 1e-5) | Dense (8192 Hidden Units) | 6 |

### Visualization

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/experiments/B2/t_acc.PNG">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/experiments/B2/t_loss.PNG">| <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/experiments/B2/v_acc.PNG">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholami/Visual-Question-Answering-with-Stacked-Attention-Networks/raw/master/experiments/B2/v_loss.PNG">|

---