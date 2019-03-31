"""description goes here

TO DO:
- add dropout to CNN
- parent class?

"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape, Concatenate
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D, Lambda
from keras.callbacks import EarlyStopping, CSVLogger # maybe more
from sklearn.naive_bayes import MultinomialNB

def embedding_layer(corpus, embed_dim, batch_size, inputs):
    return Embedding(input_dim = len(corpus.word_to_id),
                       output_dim = embed_dim,
                       weights = [corpus.init_embedding_matrix],
                       trainable = True,
                       name = "embedding")(inputs)

class LSTMSoftmax:
    """This modle is based on Khanpour et al 2016. Paper://www.aclweb.org/anthology/C16-1189
    """


    def __init__(self, corpus, batch_size = 25, num_layers = 2,
                dropout_rate = 0.5, hidden_state_size = 100, stateful = False):
        """

        Note to self, it does not sound like they passed states between batches, but why can't I try?
        """

        # this model only takes one utterance at a time, so its input shape is
        # [batch_size, utterance_length]
        inputs_ = Input(shape = (corpus.max_utt_length,), name = "input")

        # embedding layer
        embed_dim = corpus.embed_dim
        x_ = embedding_layer(corpus = corpus,
                             batch_size = batch_size, embed_dim = embed_dim,
                             inputs = inputs_)
        # shape is now [batch_size, utterance_length, embed_dim]

        # dropout layer
        x_ = Dropout(rate = dropout_rate, name = "dropout")(x_)

        # LSTM layers
        for i in range(num_layers):
            if i == 0:
                # first layer comes from embedding
                h_in_ = x_
            else:
                # subsequent inputs come from previous layers
                h_in_ = h_out_

            h_out_ = LSTM(units = hidden_state_size,
                          return_sequences = True, stateful = stateful,
                          name = "lstm_" + str(i))(h_in_)

            # output shape here is [batch_size, utterance_length, hidden_state_size]

        # pooling layer
        # they do "last pooling" -- ie, they just take the last output from the lstm layer
        lstm_out_ = Lambda(lambda x: x[:,-1,:], name = "last_pooling")(h_out_)
        # shape is now [batch_size, hidden_state_size]

        # output layer
        predictions_ = Dense(16, activation='softmax', name = 'softmax_output')(lstm_out_)

        self.model = Model(inputs = inputs_, outputs=predictions_)

    def compile(self):
        self.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')


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
        do I even need build_model??? cannot I not condense these into one function?
        what about compile??
        """

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

            # selecting utterance at a time
            u_ = Lambda(lambda x: x[:,i,:,:], name = "Lambda_" + str(i))(x_)

            # shape is now [batch_size, utt_length, embed_dim]
            u_ = Conv1D(filters = self.conv_params['filters'],
                        kernel_size = self.conv_params['kernel_size'],
                        activation='relu',
                        name = 'Conv_' + str(i))(u_)
                        # shape is now [batch_size, (utt_length - kernel_size + 1), filters]

            stv_ = GlobalMaxPooling1D(name = 'glob_max_pool_' + str(i))(u_)
            # shape is now [batch_size, filters]

            # APPLY DROPOUT HERE ???

            stv_s_.append(stv_)

        # we now construct FF layers for t-d2 ..., t-1, t
        ff_outputs = []
        for f in range(-self.d2-1,0):
            # every FF layer f takes as inputs f-d1, ... f-1, f

            if self.d1 == 0: # d1 is 0; every FF unit takes one stv as input
                s_ = stv_s_[f]

            elif self.d1 != 0 and f == -1:
                # last FF network ... take right d1+1 inputs
                s_ = stv_s_[f - self.d1:]
                s_ = Concatenate(axis = -1)(s_)
            else:
                s_ = stv_s_[f - self.d1: f+1]
                s_ = Concatenate(axis = -1)(s_)

            # shape is now [batch, filters * (d1+1)]
            ff_ = Dense(self.hidden_units, activation='relu', name = 'FF_' + str(f))(s_)
            # shape is now [batch_size, hidden_units]
            ff_outputs.append(ff_)

        # concatenate FF outputs (if necessary)
        if len(ff_outputs) == 1:
            ff_t_ = ff_outputs[0]
        else:
            ff_t_ = Concatenate(axis = -1)(ff_outputs)
        # shape is now [batch_size, (d2+1) * hidden_units]

        # ANOTHER DROPOUT?? -- they did not do that

        # output layer
        predictions_ = Dense(16, activation='softmax', name = 'softmax_output')(ff_t_)

        self.model = Model(inputs = inputs_, outputs=predictions_)

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
