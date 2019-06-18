<div align="center">
  <img src="http://uupload.ir/files/xu2_hexia.png">
  <p> © Design by Dennis Pasyuk </p>

  [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

  <img alt="Read the Docs" src="https://img.shields.io/readthedocs/hexiadocs.svg?label=Hexia%20Documentation&style=for-the-badge">
  <img alt="Codacy grade" src="https://img.shields.io/codacy/grade/62aaec49f9294a46a74c65dacf599a37.svg?color=2196F3&label=CODE%20QUALITY%20GRADE&style=for-the-badge">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/aligholami/hexia.svg?color=009688&style=for-the-badge">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/aligholami/hexia.svg?style=for-the-badge">
  <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/aligholami/hexia.svg?style=for-the-badge">
  <img alt="GitHub" src="https://img.shields.io/github/license/aligholami/hexia.svg?color=%23F44336&style=for-the-badge">
  <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/aligholami/hexia.svg?color=%23673AB7&style=for-the-badge">
</div>

----

## Introduction
This is **Hexia**. A **PyTorch** based framework for building visual question answering models. Hexia provides a mid-level API for seamless integration of your VQA models with pre-defined data, image preprocessing and natural language proprocessing pipelines.

* [`Google Slides Presentation`](https://docs.google.com/presentation/d/1KDJXSKvaUnXSl-MPtChKlgBlON9EtN3T7Oqzr0DUI5Q/edit?usp=sharing)

## Features
*   Image preprocessing
*   Text preprocessing
*   Data Handling (MS-COCO Only)
*   Real-time Loss and Accuracy Tracker
*   VQA Evaluation
*   Extendable Built-in Model Warehouse

## Installation

1. Clone the repository and enter it:

```
git clone https://github.com/aligholami/hexia && cd hexia
```

2. Run the `setup.py` to install dependencies:

```
python3 setup.py install --user
```

## Todo
- [x] Official Evaluation Support (VQA-V2)
- [x] Automatic Train/Val Plotting
- [x] Automatic Checkpointing
- [x] Automatic Resuming
- [x] Prediction Module
- [ ] Prediction Module Test
- [x] TensorboardX Auto-Resume Plots
- [ ] TensorboardX Auto-Resume Step Handler Fix
- [ ] TextVQA Support
- [ ] GQA Support
- [ ] Image Captioning Support
- [ ] Custom Loss and Optimizers

## Documentation
Checkout the full documentation [here](hexiadocs.readthedocs.io).

## References

<blockquote>1- Yang, Z., He, X., Gao, J., Deng, L., & Smola, A. (2016). Stacked attention networks for image question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 21-29). </blockquote>

<blockquote>2- Singh, A., Natarajan, V., Jiang, Y., Chen, X., Shah, M., Rohrbach, M., ... & Parikh, D. (2019). Pythia-a platform for vision & language research. In SysML Workshop, NeurIPS (Vol. 2018). </blockquote>

` More references to be added soon. `

## Contribution
Please feel free to contribute to the project. You may send a pull-request or drop me an email to talk more. ([hexpheus@gmail.com](hexpheus@gmail.com))
