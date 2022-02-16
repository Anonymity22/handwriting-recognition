# UWB-based In-air Handwriting Recognition

A machine learning model designed for UWB-based in-air handwriting recognition.

## Introduction

Using a watch equipped with UWB, users can write in the air while wearing the watch and interact with smart phone, TV and other devices for handwriting recongnition. Due to the large sensing range of UWB, it will not be restricted by the distance such as WiFi and acoustic sensing methods.  UWB-based sensing has the opportunity to support users to continuously perform handwriting in a more free manner. By designing a handwriting recognition method, the in-air handwriting interaction can be more flexible.

## Handwriting data agumentation

Below are a few samples of handwriting words collected by UWB integrated on iPhone and Apple Watch.

**![HandwritingResults](/doc/Handwriting-results.png)**

The deep neutral network needs large training dataset to obtain the high-accuracy performance. So we employ data augmentation technique to enlarge the dataset. 

Considering handwriting in different position, orientation, speed, and size, we correspondingly conduct *translation*, *rotation*, *stretch* and *scaling* operations for collected data samples. After processing with these operations, the data can be increased by six times.

Algorithm defined in [data_aug.py](/data_aug.py) can help you realize data agumentation on your own dataset. Before using this algorithm, you should put the raw handwriting data in path /data/raw_data, then enter the following command in the console.

```
python data_aug.py
```

For each input picture, you will get six enhanced picture after being processed by our algorithm. The defalut size of enhanced picture is 100\*32 pixels. 

## System overview

To achieve the free-style and user-independent handwriting recognition, we propose a recognition model to support free and continuous handwriting across different users. The model combines the recognition capability of Convolutional Recurrent Neural Network (CRNN) and the Domain-Adversarial Training of Neural Networks (DANN). The model is designed for UWB based handwriting recognition when a small number of UWB based data sets are obtained. Our model considers both recognition accuracy and generalization ability. Below is the framework of our recognition model.

**![Framework](/doc/Framework.png)**

As shown in framework, the raw data is first enhanced to form a relatively large number of training sets. In the process of continuous and cross-user recognition, one task is to construct a recognizer. When there is no letter segmentation, the recognizer can correctly identify the whole word. Another task is to construct a domain discriminator and support the cross-user capability. To enable the generalization capability, we adopt an typical handwriting dataset and use adversarial network to extract the word features independent to users. And the two tasks can be trained and learned together to achieve the optimal performance of both.

## Model design

Here, we briefly introduce the function of each module in our recognition model.

### Feature extraction

We employ CNN to design the feature extraction module. The feature extraction layer is composed of 7 layers of CNN, the dimensions of the feature extraction layer are clearly marked in the framework. The output of CNN is a  feature sequence of 25\*1\*512 size.

### Word recognition

To realize word recognition, we build a two-layer bidirectonal LSTM network to process the feature sequence. We built a two-layer bidirectional LSTM (Long Short-term Memory) network to process the feature sequence. Then we employ Connectionist Temporal Classification (CTC) layer to deal with the problem of variable-length sequence alignment.

### Domain classification

To enhance the system generalization ability on different users and unseen words, we use an adversarial netword to train the model in order to extract features only related to word content rather than users' personal writing style. In the adversarial network, a [public handwriting dataset](https://github.com/sjvasquez/handwriting-synthesis) is used as source domain, and our UWB collected dataset is used as cource domain.

## Model training

If you want to train a recognition model with your own dataset, use [model_train.py](model_train.py). Before training, make sure that you have installed Python 3.8 (Download [Here](https://www.python.org/)) and PyTorch 1.7.0 (GPU version) (Download [Here](https://pytorch.org/)). Then put the previously obtained enchanced data in path /data/train_data, and enter the following command in the console.

```
python train_crann.py
```

After model training, you can run [demo_crann.py](/rec_data.py) to test the model on the validation set.

```
python demo_crann.py
```

We offer some example UWB handwriting data in path /data/example, which can be used to train a model or validate your own model. A previously trained model is alse offered in path /model/example
