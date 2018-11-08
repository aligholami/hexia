# SAN-VQA-SEQUENCE-MODEL

### Todo

- [x] Vanilla LSTM Model
- [x] One-hot representation
- [x] One-hot training and evaluation
- [x] Embedded representation
- [ ] Embedded training and evaluation
- [ ] Embedded validation

## Hyperparamteres Analysis

I analyzed a number of different configurations for the hyperparamteres to check the effect of bigger batch size and higher learning rates on the learning of LSTM.

#### One-hot Vector Representation

##### Training and Validation Configuration #1
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

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-acc.png" alt>
    <em>Training Accuracy</em>
</p>

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-loss.png" alt>
    <em>Training Loss</em>
</p>


##### Training and Validation Configuration #2
```python
# Hyperparams
num_epochs = 500
batch_size = 20
time_steps = 1
num_features = word_vocab_size
lstm_size = 50
learning_rate = 0.01
```

##### Training and Validation Summary

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-acc-smaller-lstm-size.png" alt>
    <em>Training Accuracy</em>
</p>

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-loss-smaller-lstm-size.png" alt>
    <em>Training Loss</em>
</p>


##### Training and Validation Configuration #3
```python
# Hyperparams
num_epochs = 500
batch_size = 20
time_steps = 1
num_features = word_vocab_size
lstm_size = 50
learning_rate = 0.1
```

##### Training and Validation Summary

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-acc-smaller-rate.png" alt>
    <em>Training Accuracy</em>
</p>

<p align="center">
    <img src="https://github.com/aligholami/SAN-VQA-SEQUENCE-MODEL/raw/master/diagram/one-hot/train-loss-smaller-rate.png" alt>
    <em>Training Loss</em>
</p>
