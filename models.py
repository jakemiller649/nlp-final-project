"""description goes here"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, CSVLogger # maybe more
from sklearn.naive_bayes import MultinomialNB

def embedding_layer(corpus, embed_dim, batch_size, comb_seq_len, inputs):
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

    def __init__(self, corpus, batch_size, d1 = 0, d2 = 0):
        """
        """

        ### need some kind of multi-utterance generator
        ## gives me three at a time

        self.corpus = corpus
        self.seq_length = d1 + d2 + 1
        self.batch_size = batch_size
        self.embed_dim = corpus.embed_dim

    def build_model(self, conv_params):
        # input shape is [batch_size, seq_length, utterance_length]
        if self.seq_length == 1:
            inputs_ = Input(shape=(self.corpus.max_utt_length,), name="input")
            inputs_rs_ = inputs_
            comb_seq_len = self.corpus.max_utt_length
        else:
            inputs_ = Input(shape=(self.seq_length, self.corpus.max_utt_length), name="input")
            inputs_rs_ = Reshape((-1,) )(inputs_)
            comb_seq_len = self.seq_length * self.corpus.max_utt_length
            # essentially, we are combining a whole sequence of inputs into one big input

        # embedding
        x_ = embedding_layer(corpus = self.corpus, comb_seq_len = comb_seq_len,
                             batch_size = self.batch_size, embed_dim = self.embed_dim,
                             inputs = inputs_rs_)
        # output: [batch_size, seq_length * utt_length, embed_dim]

        # add dropout?

        # convolutional layers
        x_ = Conv1D(filters = conv_params['filters'],
                    kernel_size = conv_params['kernel_size'],
                    activation='relu')(x_)
        # shape is now [batch_size, seq_length, ??]

        stv_ = GlobalMaxPooling1D()(x_)
        #stv_ = MaxPooling1D()(x_)
        # output shape is now [batch_size, seq_length, filters]

        #stv_ = Reshape((-1))(stv_)
        # start with simple feed forward
        ff_ = Dense(64, activation='relu')(stv_)

        # this 10 is wrong, obviously
        predictions = Dense(10, activation='softmax')(ff_)

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
