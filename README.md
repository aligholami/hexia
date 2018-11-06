# SAN-VQA-SEQUENCE-MODEL

## Hyperparamteres Analysis

I analyzed a number of different configurations for the hyperparamteres to check the effect of bigger batch size and higher learning rates on the learning of LSTM.

#### One-hot Vector Representation

##### Training and Validation Configuration
```python
# Hyperparams
num_epochs = 500
batch_size = 20
time_steps = 1
num_features = word_vocab_size
lstm_size = 150
learning_rate = 0.01
```

##### Training and Validation Summary
| Train Accuracy | Train Loss |
| ------------- |:-------------:|
| <img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-acc.png"> | <img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/aligholamee/KDEPlot/blob/master/image/3_a_2.png">|
