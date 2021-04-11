# FAKE NEWS CLASSIFICATION

## RESOURCES

### TENSORFLOW COURSE ONLY ABOUT NATURAL LANGUAGE PROCESSING - https://www.coursera.org/learn/natural-language-processing-tensorflow

A whole course on learning how to deal with words, embedding, vectors, prediction and text generation using Tensorflow-based networks

Example Exercises:
Week 1 - Tokenizing Words
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-answer.ipynb
Week 2 - Classifying with FFNN https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb
Week 3 - Classifying using CNN https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb
Week 4 - Text generation using RNN https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb

### EMBEDDING PROJECTOR - https://projector.tensorflow.org/

Here you can visualize words and how they are "understood" from your trained model. It can help to visualize the data and see if your model learned correctly some pattern and correlation within words.

Here the notebook to generate a file to upload there
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb
(it's part of the Coursera Course Natural Language Processing on Tensorflow - https://www.coursera.org/learn/natural-language-processing-tensorflow)


## EXAMPLES WITH RNNs

### EXAMPLE RNN - https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

Nice tutorials, it explains step by step what to do and why

Differences:

- Tokenizer parameters (but you already have done it)
- Y_train has 13 values (one hot encoding of 13 labels), in our case only 1 value (binary encoding)
- Softmax is not needed, better sigmoid
- it uses dropout, maybe at the beginning is not needed
- loss function should be change to binary crossentropy
- adam optimizer sometimes needs learning rate (usually between 1e-3 and 1e-5)


### EXAMPLE TENSORFLOW - https://www.tensorflow.org/tutorials/text/text_classification_rnn

Tutorial using keras. It does not explain too much in details, it can be interesting but confusing

Differences:

- dataset and preparation (they use the API, you do not need)
- model (it uses the keras notation instead of the tensorflow one, it may be confusing)

## EXAMPLE ON KAGGLE - https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

Similar example, done in a very quick way (classifying spam mails), not too much explanation, but a couple of useful stuff

Differences:

- dataset, they use LabelEncoder from sklearn. Not needed for you
- More complex model, I think in the beginning is also not very important.

## EXAMPLE ON MACHINELEARNINGMASTERY - https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

A longer tutorial with different examples. Quite well-written and adding difficulties at each step.

