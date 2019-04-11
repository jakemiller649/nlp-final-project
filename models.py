"""
This file includes the models for two of my three models (plus a Naive Bayes). It contains:
- LSTM-Softmax (based on Khanpour et al 2016)
- CNN (based on Lee and Dernoncourt 2016)

"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape, Concatenate, CuDNNLSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D, Lambda
from sklearn.naive_bayes import MultinomialNB


def embedding_layer(corpus, embed_dim, inputs, trainable):
    """
    Common embedding layer to be used by all models.
    Args:
      - embed_dim: this is predetermined based on what pretraiend word embeddings we are using
      - corpus: This the corpus we are training the model on, and it provides the initial embedding matrix
        (see utils file for how this is developed) and the input dimension
      - inputs: input vector from training data (must be output from keras Input() layer)
            - this is either [batch_size, utterance_length] or [batch_size, sequence_length, utterance_length]
              depending on which model is running
      - trainable: boolean as to whether our pretrained word embeddings will be updated during model training
    Returns:
      - [batch_size, utterance_length, embed_dim] or [batch_size, sequence_length, utterance_length, embed_dim]
    """
    
    return Embedding(input_dim = len(corpus.word_to_id),
                       output_dim = embed_dim,
                       weights = [corpus.init_embedding_matrix],
                       trainable = trainable,
                       name = "embedding")(inputs)

def layered_LSTM(x_, num_layers, hidden_state_size, stateful, bidirectional = False, suffix = ""):
    
    """
    Creates a stack of LSTM layers
    Args:
     - x_: inputs from previous layer
     - num_layers: number of LSTM layers
     - hidden_state_size: dim of hidden state size
     - stateful: from keras docs: boolean the last state for each sample at 
                 index i in a batch will be used as initial state 
                 for the sample of index i in the following batch
     - bidirectional: boolean as to whether the LSTM layers are bidirectional
     - suffix: a string that is incremented if necessary for naming layers
     """

    for i in range(num_layers):
        if i == 0:
            # first layer comes from embedding
            h_in_ = x_
        else:
            # subsequent inputs come from previous layers
            h_in_ = h_out_

        if bidirectional == False:
            h_out_ = CuDNNLSTM(units = hidden_state_size,
                               return_sequences = True, stateful = stateful,
                               name = "lstm_" + suffix + str(i))(h_in_)
        elif bidirectional == True:
            h_out_ = Bidirectional(CuDNNLSTM(units = hidden_state_size,
                                             return_sequences = True, stateful = stateful,
                                             name = "bilstm_" + str(i)),
                                   merge_mode = 'concat')(h_in_)

    return h_out_


class LSTMSoftmax:
    """This model is based on Khanpour et al 2016. Paper://www.aclweb.org/anthology/C16-1189
    """


    def __init__(self, corpus, batch_size = 25, num_layers = 2,
                dropout_rate = 0.5, hidden_state_size = 100, stateful = False,
                bidirectional = False, trainable_embed = False):
        """
        Args:
         - corpus: the AMI_Corpus object (see utils)
         - batch_size: batch_size for training
         - num_layers: number of LSTM layers
         - dropout_rate: rate for dropout layers
         - hidden_state_size: dim of hidden state size
         - stateful: from keras docs: boolean the last state for each sample at 
                     index i in a batch will be used as initial state 
                     for the sample of index i in the following batch
         - bidirectional: boolean as to whether the LSTM layers are bidirectional
         - trainable_embed: whether we want our embedding layer (which has pretrained word vectors) to be updated
           during training
        """

        # this model only takes one utterance at a time, so its input shape is
        # [batch_size, utterance_length]
        inputs_ = Input(shape = (corpus.max_utt_length,), name = "input")

        # embedding layer
        embed_dim = corpus.embed_dim
        x_ = embedding_layer(corpus = corpus,
                             embed_dim = embed_dim,
                             inputs = inputs_, trainable = trainable_embed)
        # shape is now [batch_size, utterance_length, embed_dim]

        # dropout layer
        x_ = Dropout(rate = dropout_rate, name = "dropout")(x_)

        # LSTM layers
        h_out_ = layered_LSTM(x_ = x_, num_layers = num_layers,
                              hidden_state_size = hidden_state_size, stateful = stateful,
                              bidirectional = bidirectional)
        # output shape here is [batch_size, utterance_length, hidden_state_size]
        # unless bidirectional is true, in which case last dim is 2*hidden_state_size

        # pooling layer
        # they do "last pooling" -- ie, they just take the last output from the lstm layer
        lstm_out_ = Lambda(lambda x: x[:,-1,:], name = "last_pooling")(h_out_)
        # shape is now [batch_size, hidden_state_size]

        # output layer
        predictions_ = Dense(17, activation='softmax', name = 'softmax_output')(lstm_out_)

        self.model = Model(inputs = inputs_, outputs=predictions_)

    def compile(self):
        self.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')


class CNN:
    """
    Model is based on Lee and Dernoncourt 2016: Sequential Short-Text Classification with
    Recurrent and Convolutional Neural Networks.
    https://arxiv.org/pdf/1603.03827.pdf

    Their best results came from their CNN so that's what I am focusing on here.

    Two important hyperparameters: d1 and d2, which denote 'history sizes'. See my paper (or theirs, I guess
    for full explanation). 
    """

    def __init__(self, corpus, batch_size, filters, kernel_size,
                 hidden_units, dropout_rate = 0, d1 = 0, d2 = 0, trainable_embed = False):
        """
        Args:
         - corpus: the AMI_Corpus object (see utils)
         - batch_size: batch_size for training
         - filters: number of convolutional filters
         - kernel_size: kernel size of filters
         - hidden_units: size of feed-forward layers
         - dropout_rate: rate for dropout layers
         - d1, d2: history parameters
         - trainable_embed: whether we want our embedding layer (which has pretrained word vectors) to be updated
           during training
        """

        self.corpus = corpus
        seq_length = d1 + d2 + 1
        embed_dim = corpus.embed_dim

        # input shape is [batch_size, seq_length, utterance_length]
        inputs_ = Input(shape=(seq_length, self.corpus.max_utt_length), name="input")

        # embedding
        x_ = embedding_layer(corpus = self.corpus,
                             embed_dim = embed_dim,
                             inputs = inputs_, trainable = trainable_embed)
        # output: [batch_size, seq_length, utt_length, embed_dim]

        # convolutional layers
        stv_s_ = []
        for i in range(seq_length):

            # selecting utterance at a time
            u_ = Lambda(lambda x: x[:,i,:,:], name = "Lambda_" + str(i))(x_)

            # shape is now [batch_size, utt_length, embed_dim]
            u_ = Conv1D(filters = filters,
                        kernel_size = kernel_size,
                        activation='relu',
                        name = 'Conv_' + str(i))(u_)
                        # shape is now [batch_size, (utt_length - kernel_size + 1), filters]

            stv_ = GlobalMaxPooling1D(name = 'glob_max_pool_' + str(i))(u_)
            # shape is now [batch_size, filters]

            # Lee and Dernoncourt applied dropout here
            stv_ = Dropout(rate = dropout_rate, name = "dropout_" + str(i))(stv_)
            stv_s_.append(stv_)

        # we now construct FF layers for t-d2 ..., t-1, t
        ff_outputs = []
        for f in range(-d2-1,0):
            # every FF layer f takes as inputs f-d1, ... f-1, f

            if d1 == 0: # d1 is 0; every FF unit takes one stv as input
                s_ = stv_s_[f]

            elif d1 != 0 and f == -1:
                # last FF network ... take right d1+1 inputs
                s_ = stv_s_[f - d1:]
                s_ = Concatenate(axis = -1)(s_)
            else:
                s_ = stv_s_[f - d1: f+1]
                s_ = Concatenate(axis = -1)(s_)

            # shape is now [batch, filters * (d1+1)]
            ff_ = Dense(hidden_units, activation='relu', name = 'FF_' + str(f))(s_)
            # shape is now [batch_size, hidden_units]
            ff_outputs.append(ff_)

        # concatenate FF outputs (if necessary)
        if len(ff_outputs) == 1:
            ff_t_ = ff_outputs[0]
        else:
            ff_t_ = Concatenate(axis = -1)(ff_outputs)
        # shape is now [batch_size, (d2+1) * hidden_units]

        # output layer
        predictions_ = Dense(17, activation='softmax', name = 'softmax_output')(ff_t_)

        self.model = Model(inputs = inputs_, outputs=predictions_)

    def compile(self):
        self.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')


class NaiveBayes():
    """As a sanity check, we are seeing what Naive Bayes would give score."""

    def __init__(self, corpus, ngram_range = (1,1), tfidf = False):
        """
        Initialize model class.
        
        Args: 
          - corpus. The corpus we are working with
          - ngram_range: Whether we want bigrams, unigrams, etc. Tuple of length two - upper and lower bounds
            on ngram length
          - tfidf: boolean as to whether vectorize each utterance with simple word counts or with tfidf
         """
        
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        if tfidf is True:
            vectorizer = TfidfVectorizer(ngram_range = ngram_range)
        else:
            vectorizer = CountVectorizer(ngram_range = ngram_range)

        train_utterances = []
        for c in corpus.train_convos:
            for u in c.utterances:
                train_utterances.append(u)

        test_utterances = []
        for c in corpus.test_convos:
            for u in c.utterances:
                test_utterances.append(u)

        x = [" ".join(u.words) for u in train_utterances if u.da_type is not None]
        self.y_train = [u.da_type for u in train_utterances if u.da_type is not None]

        # create term-doc matrix for training
        self.x_train = vectorizer.fit_transform(x)

        # create for test
        test_words = [" ".join(u.words) for u in test_utterances if u.da_type is not None]
        self.x_test = vectorizer.transform(test_words)
        self.y_test = [u.da_type for u in test_utterances if u.da_type is not None]

    def random_search(self, cv = 5, n_iter = 20, param_dist = None):
        """ Conducts a random search of the NB smoothing parameter (alpha)"""
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform

        if param_dist is None:
            param_dist = {'alpha':uniform(loc = 0, scale = 5)}

        mnb = MultinomialNB()
        rscv = RandomizedSearchCV(estimator = mnb, param_distributions = param_dist,
                                  cv = cv, n_iter = n_iter, scoring = 'accuracy',
                                  return_train_score = False)
        rscv.fit(self.x_train, self.y_train)
        return rscv.cv_results_

    def train(self, alpha = 1):
        """Train Multinomial NB with alpha parameter"""
        self.mnb = MultinomialNB(alpha = alpha)
        self.mnb.fit(self.x_train, self.y_train)

    def eval_on_test(self):
        """Evaluate on test data"""
        from sklearn.metrics import accuracy_score, log_loss
        pred = self.mnb.predict(self.x_test)
        acc = accuracy_score(self.y_test, pred)
        return acc
