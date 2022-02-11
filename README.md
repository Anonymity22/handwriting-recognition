# UWB-based In-air Handwriting Recognition

A machine learning model designed for UWB-based in-air handwriting recognition.

## Introduction

Using a watch equipped with UWB, users can write in the air while wearing the watch and interact with smart phone, TV and other devices for handwriting recongnition. Due to the large sensing range of UWB, it will not be restricted by the distance such as WiFi and acoustic sensing methods.  UWB-based sensing has the opportunity to support users to continuously perform handwriting in a more free manner. By designing a handwriting recognition method, the in-air handwriting interaction can be more flexible.

## Handwriting data agumentation

Below are a few samples of handwriting words collected by UWB integrated on iPhone and Apple Watch.

**![HandwritingResults](/doc/Handwriting-results.png)**

The deep neutral network needs large training dataset to obtain the high-accuracy performance. So we employ data augmentation technique to enlarge the dataset. 

Considering handwriting in different position, orientation, speed, and size, we correspondingly conduct *translation*, *rotation*, *stretch* and *scaling* operations for collected data samples. After processing with these operations, the data can be increased by six times.

Algorithm in [data_aug.py](/data_aug.py) can help you realize data agumentation on your own dataset.

### Usage:
```
python data_aug.py
```

For each input picture, you will get six enhanced picture after being processed by our algorithm. The size of each enhanced picture is 100*32 pixels. 

## System overview

To achieve the free-style and user-independent handwriting recognition, we propose a recognition model to support free and continuous handwriting across different users. The model combines the recognition capability of Convolutional Recurrent Neural Network (CRNN) and the Domain-Adversarial Training of Neural Networks (DANN). The model is designed for UWB based handwriting recognition when a small number of UWB based data sets are obtained. Our model considers both recognition accuracy and generalization ability. Below is the framework of our recognition model.

**![Framework](/doc/Framework.png)**

As shown in framework, the raw data is first enhanced to form a relatively large number of training sets. In the process of continuous and cross-user recognition, one task is to construct a recognizer. When there is no letter segmentation, the recognizer can correctly identify the whole word. Another task is to construct a domain discriminator and support the cross-user capability. To enable the generalization capability, we adopt an typical handwriting dataset and use adversarial network to extract the word features independent to users. And the two tasks can be trained and learned together to achieve the optimal performance of both.

## Model design

Here, we briefly introduce the function of each module in our recognition model.

### Feature extraction

We employ CNN to design the feature extraction module. The feature extraction layer is composed of 7 layers of CNN, the dimensions of the feature extraction layer are clearly marked in the framework. The output of CNN is a sequence

### Word recognition

To realize word recognition, we build a two-layer bidirectonal LSTM network to process the feature sequence. The goal of LSTM is to predict which character this

Based on the feature extraction module, we notice that the feature sequence extracted by CNN corresponds to different positions of word. This means that a feature sequence may correspond to a part of a letter, and adjacent feature sequences can also reflect the difference between adjacent letters. To recognize words, we built a two-layer bidirectional LSTM (Long Short-term Memory) network to process the feature sequence corresponding to the probability vector on different time. The goal of LSTM is to predict which character this rectangular area is, that is, to predict according to the feature vector extracted by the multi-layer CNN, and obtain the softmax probability distribution of all letters.

Then we employ Connectionist Temporal Classification (CTC) layer~\cite{} to deal with the problem of variable-length sequence alignment. The CTC loss function is obtained to carry out end-to-end joint training. For example, if there are 6 time windows while recognizing handwriting, ideally, t0, t1, and t2 should be mapped to "u", t3, t4 should be mapped to "b", and t5 should be mapped to "i". And then concatenating these character sequences to get "uuubbi", we merge the consecutive repeated characters into one, and the final result is "ubi". When the word itself contains repeated letters (such as hello), CTC solves this problem by introducing a blank character. The character prediction sequence (such as “-hh-el-ll-o–”) can be obtained according to the probability vector. At last, we use HMM’s Forward-Backward algorithm and dynamic programming algorithm to calculate
the probability. Thus, the CTC loss function ($L_p$) is calculated using cross entropy with word label.

### Domain classification

To enhance the system generalization ability on different users and unseen words, we need to extract features independently only related to word content and cross different users, that is, when a new user comes, the handwriting can be directly recognized. We introduce a adversarial network to train the model. The public dataset is used as source domain, which contains handwriting words with different writing style. Our UWB collected dataset is the target domain. We design the domain classifier so that it cannot identify whether it is the word from the public data set or the word from UWB data set. This ensures that the feature extraction is independent with the writing style between users. The loss function of domain classification ($L_d$) is also calculated using cross entropy with domain label.

## Usage
