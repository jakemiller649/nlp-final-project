"""description goes here"""

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, CSVLogger # maybe more
from sklearn.naive_bayes import MultinomialNB

class BiLSTMSoftmax:
    # eventually change name
    """model pased on Khanpour et al """

    def __init__(self, corpus, base, output, **params):
        """ xxx """

        pass


class CNN:
    """yada yada yada"""

    def __init__(self, corpus):
        pass

class BiLSTMCRF:
    """Model based on Kumar et al 2018, the current high score on the
       switchboard corpus."""

    def __init__(self, corpus):

        pass

class NaiveBayes():
    """In order to have a baseline to compare against."""

    def __init__(self, corpus, tfidf = False, **kwargs):
        pass
