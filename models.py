"""description goes here"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape, Concatenate
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, CSVLogger # maybe more
from sklearn.naive_bayes import MultinomialNB

def embedding_layer(corpus, embed_dim, batch_size, inputs):
    return Embedding(input_dim = len(corpus.word_to_id),
                       output_dim = embed_dim,
                       weights = [corpus.init_embedding_matrix],
                       trainable = True)(inputs)

class BiLSTMSoftmax:
    # eventually change name
    """model pased on Khanpour et al """

    def __init__(self, corpus, base, output, **params):
        """ xxx """

        pass


class CNN:
    """
    Model is based on Lee and Dernoncourt 2016: Sequential Short-Text Classification with
    Recurrent and Convolutional Neural Networks.
    https://arxiv.org/pdf/1603.03827.pdf

    Their best results came from their CNN so that's what I am focusing on here.

    Two important hyperparameters: d1 and d2, which denote 'history sizes'.

    Longer explanation goes here.
    """

    def __init__(self, corpus, batch_size, conv_params, hidden_units, d1 = 0, d2 = 0):
        """
        """

        ### need some kind of multi-utterance generator
        ## gives me three at a time

        self.corpus = corpus
        self.seq_length = d1 + d2 + 1
        self.batch_size = batch_size
        self.embed_dim = corpus.embed_dim
        self.conv_params = conv_params
        self.hidden_units = hidden_units
        self.d1 = d1
        self.d2 = d2

    def build_model(self):

        # input shape is [batch_size, seq_length, utterance_length]
        inputs_ = Input(shape=(self.seq_length, self.corpus.max_utt_length), name="input")

        # embedding
        x_ = embedding_layer(corpus = self.corpus,
                             batch_size = self.batch_size, embed_dim = self.embed_dim,
                             inputs = inputs_)
        # output: [batch_size, seq_length, utt_length, embed_dim]

        # convolutional layers
        stv_s_ = []
        for i in range(self.seq_length):
            u_ = x_[:,i,:,:] # selecting utterance at a time
            # shape is now [batch_size, utt_length, embed_dim]
            u_ = Conv1D(filters = self.conv_params['filters'],
                        kernel_size = self.conv_params['kernel_size'],
                        activation='relu')(u_)
                        # shape is now [batch_size, (utt_length - kernel_size + 1), filters]

            stv_ = GlobalMaxPooling1D()(u_)
            # shape is now [batch_size, filters]

            # APPLY DROPOUT HERE ???

            stv_s_.append(stv_)

        # we now construct FF layers for t-d2 ..., t-1, t
        ff_outputs = []
        for f in range(-self.d2-1,0):
            # every FF layer f takes as inputs f-d1, ... f-1, f
            s_ = stv_s_[f - self.d1: f+1]
            s_ = Concatenate(axis = -1)(s_)
            # shape is now [batch, filters * (d1+1)]
            ff_ = Dense(self.hidden_units, activation='relu')(s_)
            # shape is now [batch_size, hidden_units]

        # concatenate FF outputs
        ff_t_ = Concatenate(axis = -1)
        # shape is now [batch_size, (d2+1) * hidden_units]

        # ANOTHER DROPOUT??

        # output layer
        predictions = Dense(16, activation='softmax')(ff_t_)

        self.model = Model(inputs = inputs_, outputs=predictions)

    def compile(self):
        self.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')



class BiLSTMCRF:
    """Model based on Kumar et al 2018, the current high score on the
       switchboard corpus."""

    def __init__(self, corpus):

        pass

class NaiveBayes():
    """In order to have a baseline to compare against."""

    def __init__(self, corpus, tfidf = False, **kwargs):
        pass
