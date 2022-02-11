# In-air Handwriting Recognition

A machine learning model designed for UWB-based in-air handwriting recognition.

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

## Data process and augmentation
The deep neutral network needs large training dataset to obtain the high-accuracy performance. However, it is difficult to manually collect large number of handwriting samples in practice. We employ data augmentation technique to enlarge the dataset. Considering handwriting in different position, orientation, speed, and size, we correspondingly conduct \textit{translation}, \textit{rotation}, \textit{stretch} and \textit{scaling} operations for collected data samples. Note that user performs handwriting in 3D-space, we process the UWB collected handwriting trajectory data through 3D-mapping as in Section 4 and thus compress to the 2D plane. After processing with these operations, the data can be increased. Besides, to further improve model generalization ability, we design to combine the existing handwritten dataset~\cite{Handwriting} to enhance the training set. We integrate a commonly used generated handwriting dataset, which includes 8 different writing styles. 

**![Framework](/doc/Framework.png)**

## System Overview
To achieve the free-style and user-independent handwriting recognition, we propose a recognition model to support free and continuous handwriting across different users. The model combines the recognition capability of Convolutional Recurrent Neural Network (CRNN) and the Domain-Adversarial Training of Neural Networks (DANN). The model is designed for UWB based handwriting recognition when a small number of UWB based data sets are obtained. Our model considers both recognition accuracy and generalization ability. As shown in [our system design](/doc/Framework.png), based on the handwriting data collected by UWB, the word is constructed through 3D-projection and trajectory filtering. Then the data are enhanced to form a relative large number of  training sets. In the process of continuous and cross-user recognition, one task is to construct a recognizer. When there is no letter segmentation, the recognizer can correctly identify the whole word. Another task is to construct a domain discriminator and support the cross-user capability. To enable the generalization capability, we adopt an typical handwriting dataset and use adversarial network to extract the word features independent to users. And the two tasks can be trained and learned together to achieve the optimal performance of both.**

## Model design
\subsection{Model design}

**Feature extraction** We employ CNN to design the feature extraction module. The feature extraction layer is composed of 7 layers of CNN, the dimensions of the feature extraction layer are marked in the Figure~\ref{fig:Neural}. This layer is used to extract feature sequence from the handwriting data. Since the convolutional neural network has translation invariance for feature extraction, it can be seen from Figure~\ref{fig:Neural} that each column of the feature sequence corresponds to a specific area, and these areas correspond to the word with same order from left to right. The feature  sequence extracted from the convolutional layer is used as the input of the world recognition module and domain classification module.

**Word recognition** Based on the feature extraction module, we notice that the feature sequence extracted by CNN corresponds to different positions of word. This means that a feature sequence may correspond to a part of a letter, and adjacent feature sequences can also reflect the difference between adjacent letters. To recognize words, we built a two-layer bidirectional LSTM (Long Short-term Memory) network to process the feature sequence corresponding to the probability vector on different time. The goal of LSTM is to predict which character this rectangular area is, that is, to predict according to the feature vector extracted by the multi-layer CNN, and obtain the softmax probability distribution of all letters.

Then we employ Connectionist Temporal Classification (CTC) layer~\cite{} to deal with the problem of variable-length sequence alignment. The CTC loss function is obtained to carry out end-to-end joint training. For example, if there are 6 time windows while recognizing handwriting, ideally, t0, t1, and t2 should be mapped to "u", t3, t4 should be mapped to "b", and t5 should be mapped to "i". And then concatenating these character sequences to get "uuubbi", we merge the consecutive repeated characters into one, and the final result is "ubi". When the word itself contains repeated letters (such as hello), CTC solves this problem by introducing a blank character. The character prediction sequence (such as “-hh-el-ll-o–”) can be obtained according to the probability vector. At last, we use HMM’s Forward-Backward algorithm and dynamic programming algorithm to calculate
the probability. Thus, the CTC loss function ($L_p$) is calculated using cross entropy with word label.


**Domain classification** To enhance the system generalization ability on different users and unseen words, we need to extract features independently only related to word content and cross different users, that is, when a new user comes, the handwriting can be directly recognized. We introduce a adversarial network to train the model. The public dataset is used as source domain, which contains handwriting words with different writing style. Our UWB collected dataset is the target domain. We design the domain classifier so that it cannot identify whether it is the word from the public data set or the word from UWB data set. This ensures that the feature extraction is independent with the writing style between users. The loss function of domain classification ($L_d$) is also calculated using cross entropy with domain label.
