# In-air Handwriting Recognition

A learning model for UWB-based in-air handwriting recognition

## Introduction

[Connectionist Temporal Classification](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
is a loss function useful for performing supervised learning on sequence data,
without needing an alignment between input data and labels.  For example, CTC
can be used to train
[end-to-end](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
[systems](http://arxiv.org/pdf/1408.2873v2.pdf) for
[speech recognition](http://arxiv.org/abs/1512.02595),
which is how we have been using it at Baidu's Silicon Valley AI Lab.

![DSCTC](/doc/deep-speech-ctc-small.png)

The illustration above shows CTC computing the probability of an output
sequence "THE CAT ", as a sum over all possible alignments of input sequences
that could map to "THE CAT ", taking into account that labels may be duplicated
because they may stretch over several time steps of the input data (represented by
the spectrogram at the bottom of the image).
Computing the sum of all such probabilities explicitly would be prohibitively costly due to the
combinatorics involved, but CTC uses dynamic programming to dramatically
reduce the complexity of the computation. Because CTC is a differentiable function,
it can be used during standard SGD training of deep neural networks.

In our lab, we focus on scaling up recurrent neural networks, and CTC loss is an
important component. To make our system efficient, we parallelized the CTC
algorithm, as described in [this paper](http://arxiv.org/abs/1512.02595).
This project contains our high performance CPU and CUDA versions of the CTC loss,
along with bindings for [Torch](http://torch.ch/).
The library provides a simple C interface, so that it is easy to
integrate into deep learning frameworks.

This implementation has improved training scalability beyond the
performance improvement from a faster parallel CTC implementation. For
GPU-focused training pipelines, the ability to keep all data local to
GPU memory allows us to spend interconnect bandwidth on increased data
parallelism.

## System Overview
To achieve the free-style and user-independent handwriting recognition, we propose a recognition model to support free and continuous handwriting across different users. The model combines the recognition capability of Convolutional Recurrent Neural Network (CRNN) and the Domain-Adversarial Training of Neural Networks (DANN). The model is designed for UWB based handwriting recognition when a small number of UWB based data sets are obtained. Our model considers both recognition accuracy and generalization ability. As shown in [our system design](/doc/framework), based on the handwriting data collected by UWB, the word is constructed through 3D-projection and trajectory filtering. Then the data are enhanced to form a relative large number of  training sets. In the process of continuous and cross-user recognition, one task is to construct a recognizer. When there is no letter segmentation, the recognizer can correctly identify the whole word. Another task is to construct a domain discriminator and support the cross-user capability. To enable the generalization capability, we adopt an typical handwriting dataset and use adversarial network to extract the word features independent to users. And the two tasks can be trained and learned together to achieve the optimal performance of both.**
