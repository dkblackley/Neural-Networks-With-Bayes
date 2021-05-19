
# Cost and Risk aware Skin lesion Classification using Bayesian Inference

This repository is the official technical manual of my fourth year dissertation, comparing MC Dropout (https://arxiv.org/abs/1506.02142) to Bayes by Backprop (https://arxiv.org/abs/1505.05424) to a standard Softmax Response.


### Requirements

To install libraries:

```setup
pip install -r requirements.txt
```

After requirements have been installed navigate into the python directory.

Download the ISIC 2019 test data at the following links: 
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip

extract the training images into python/ISIC\_2019\_Training_Input and the test input into python/ISIC\_2019\_Test\_Input respectively, remove any non image files, such as the metadata, license etc...

# Project Abstract

Skin lesions are areas of the skin that have abnormal growths and can be an indi­cation of skin cancer, thankfully, these can usually be identified with a visual exami­nation. Manual examination is a rather te­dious and time consuming job for expert der­matologists, recent advances in deep learn­ing has helped automate the classification of skin lesions while showing remarkable ac­curacy. These tools do not necessarily take into account the cost of miss-­classifying a life threatening lesion with a benign one how­ever.We employed three machine learning algorithms:  A standard softmax baseline, Monte Carlo Dropout and Bayes by Back­prop in a cost ­aware environment and com­pared the results. We found that when com­pared solely with accuracy MC Dropout per­forms best, but only slightly better than a softmax response.  When we employ our model in a cost aware setting Bayes by Back­prop performs best.
## Methods used

### Softmax response
### Bayes by Backprop
### MC dropout

## Evaluation metrics

### Accuracy
### Reliability
### Cost

## Usage

To train a model with default settings, run this command:

```train
python python/main.py
```

We also have several optional parameters to pass to the main file
```train
python python/main.py -e50 -fp100 -bbb -isic -cpu -load -predict -n3
```
Use -e<number_of_epochs> to determine how many epochs to train for. Default is 100.

```train
python python/main.py -fp100
```
Use -fp<number_of_forward_passes> To determine how many forward passes to run thourgh our Bayesian Neural Networks

```train
python python/main.py -bbb
```
Use -bbb to train Bayes by Backprop network

```train
python python/main.py -isic
```
Use -isic to predict on the ISIC 2019 test set

```train
python python/main.py -cpu
```
Use -cpu to train using a CPU

```train
python python/main.py -load
```
Use -load to train a model from python/saved\_models/BBB\_Classifier\_0 or python/saved\_models/SM\_Classifier\_0 if using a softmax classifier.

```train
python python/main.py -predict
```
Use -predict to skip training and go straight to evaluation

```train
python python/main.py -n3
```
Use -n<number_of_models> to train multiple models


# Results

We found that with the following parameters we get these results on our generated risk coverage curve:


| Method            | Accuracy | Acc. Covg, AUC | Avg. Test Cost | Cost Covg. AUC |
|-------------------|----------|----------------|----------------|----------------|
| SR                | 79.69%   | 0.92           | 4.81           | 2.451          |
| MC Dropout        | 73.73%   | 0.925          | 4.53           | 2.362          |
| Bayes by Backprop | 78.96%   | 0.919          | 4.35           | 1.963          |


We also obtained 150th place as of April 27th 2021 on the ISIC 2019 leaderboard.


