"""
This file contains the model class for the Bi-LSTM-CRF. It's separate because all of its parts and pieces are 
from regular keras instead of tf.keras. This is so it works with the CRF layer from keras_contrib.
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape, Concatenate, CuDNNLSTM
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D, Lambda


def embedding_layer(corpus, embed_dim, batch_size, inputs, trainable):
    return Embedding(input_dim = len(corpus.word_to_id),
                       output_dim = embed_dim,
                       weights = [corpus.init_embedding_matrix],
                       trainable = trainable,
                       name = "embedding")(inputs)

def layered_LSTM(x_, num_layers, hidden_state_size, stateful, bidirectional = False, suffix = ""):

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

class BiLSTMCRF:
    """Model based on Kumar et al 2018, the current high score on the
       switchboard corpus.

       This is different than the previous models in that it takes whole conversations
       as inputs.
       """

    def __init__(self, corpus, batch_size = 25, sequence_length = 50, dropout_rate = 0.0, bidirectional = True, num_layers = 1,
                 hidden_state_size = 100, stateful = False, trainable_embed = False):

        from keras_contrib.layers import CRF

        inputs_ = Input(shape = (sequence_length, corpus.max_utt_length), name = "input")

        # embedding layer
        embed_dim = corpus.embed_dim
        x_ = embedding_layer(corpus = corpus,
                             batch_size = batch_size, embed_dim = embed_dim,
                             inputs = inputs_, trainable = trainable_embed)
        # shape is now [batch_size, convo_length, utterance_length, embed_dim]

        # dropout layer
        x_ = Dropout(rate = dropout_rate, name = "dropout")(x_)

        ### UTTERANCE-LEVEL ENCODER ####
        encoded_utterances = []

        # Each utterance gets its own encoder that consists of LSTM and pooling
        for i in range(sequence_length):
            # selecting utterance at a time
            u_ = Lambda(lambda x: x[:,i,:,:], name = "Extraction_" + str(i))(x_)

            # LSTM
            h_out_ = layered_LSTM(x_ = u_, num_layers = num_layers,
                                  hidden_state_size = hidden_state_size, stateful = stateful,
                                  bidirectional = bidirectional, suffix = 'a_')
            # output shape here is [batch_size, utterance_length, hidden_state_size]
            # if bidirectional is true, in which case last dim is 2*hidden_state_size

            # Pooling
            # we are last pooling by each utterance
            lstm_out_ = Lambda(lambda x: x[:,-1,:], name = "last_pooling"+ str(i))(h_out_)
            # size here is now [batch_size, hidden_state_size]
            encoded_utterances.append(lstm_out_)

        #### CONVERSATION-LEVEL ENCODER ###
        #### CONVERSATION-LEVEL ENCODER ###
        if len(encoded_utterances) > 1:
            conv_ = Concatenate(axis = 1)(encoded_utterances)
            # shape is now [batch_size, convo_len*hidden_state_size]
     
        else:
            conv_ = encoded_utterances[0]
        
        conv_ = Reshape((sequence_length, -1))(conv_)
        # shape is now [batch_size, convo_len, hidden_state_size]

        c_out_ = layered_LSTM(x_ = conv_, num_layers = num_layers,
                              hidden_state_size = hidden_state_size, stateful = stateful,
                              bidirectional = bidirectional, suffix = 'b_')

        ### LINEAR CHAIN CRF OUTPUT LAYER ##

        predictions_ = CRF(17)(c_out_)

        self.model = Model(inputs = inputs_, outputs = predictions_)


    def compile(self):
        from keras_contrib.losses import crf_loss
        from keras_contrib.metrics import crf_viterbi_accuracy

        self.model.compile(optimizer = 'adagrad', metrics = [crf_viterbi_accuracy], loss = crf_loss)

        
from keras.utils import Sequence       
   
class UtteranceGenerator(Sequence):
    """
    Inherits Keras Sequence class to optimize multiprocessing
    doc string goes here"""

    def __init__(self, corpus, mode, batch_size, sequence_length = 1, algo = None):
        """
        Args
        Returns
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.algo = algo
        
        assert self.sequence_length >= 1
        
        if algo not in ["LSTM_Soft", "CNN", "LSTM_CRF"]:
            raise Exception("Algo must be one of [\"LSTM_Soft\", \"CNN\", \"LSTM_CRF\"]")

        if mode == "train":
            convos = corpus.train_convos
        elif mode == "val":
            convos = corpus.val_convos
        elif mode == "test":
            convos = corpus.test_convos
        else:
            raise Exception("Generator's mode must be either 'train,' 'val,' or 'test'")

        # create universal indexing for utterances
        self.combined_utterances = []
        for c in convos:
            for u in c.utterances:
                self.combined_utterances.append(u)

    def __getitem__(self, idx):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        start = idx * self.batch_size 
        end = (idx + 1) * self.batch_size 
        
        if end > len(self.combined_utterances) - self.sequence_length: # end cases
            end = len(self.combined_utterances) - self.sequence_length

        if self.algo is "LSTM_Soft":
            batch_x = np.array([u.word_ids for u in self.combined_utterances[start:end]])
            # batch x shape is [batch_size, utterance_length]
        else:
            if self.algo is "LSTM_CRF":
                # the LSTM_CRF algorithm guesses on each utterance in a sequence, so 
                # we step over the data in sequence_length steps
                step = self.sequence_length
            else:
                # whereas the CNN, even if it takes t-1, t-2, etc., as input, only guesses about 
                # the utterance at t, so we only step forward one at a time
                step = 1
            
            sub_batches_x = []
            for idx in range(start,end, step):
                sub_batch_x = np.array([u.word_ids for u in self.combined_utterances[idx:idx+self.sequence_length]])
                sub_batches_x.append(sub_batch_x)
            batch_x = np.array(sub_batches_x)
            # batch x shape is [batch_size, sequence_length, utterance_length]

        ## moving on to Y batches
        if self.algo is not "LSTM_CRF":
            # regular LSTM yield only one label at a time
            # # CNN also yields one, but it is at end of sequence
            sm1 = self.sequence_length - 1
            
            batch_y = np.array([u.da_type for u in self.combined_utterances[start + sm1:end + sm1]], dtype = np.int)
            # batch_y shape is currently [batch,]

            # convert batch_y to one-hot
            batch_y_oh = np.zeros((batch_y.shape[0], 17))
            batch_y_oh[np.arange(batch_y.shape[0]), batch_y] = 1
            # batch_y_oh is [batch_size, num_classes]
            
        else:
            # The LSTM-CRF expects one label per utterance, so a sequence length of 5 should yield 5 labels
            sub_batches_y = []
            for idx in range(start,end, step):
                sub_batch_y = np.array([u.da_type for u in self.combined_utterances[idx:idx+self.sequence_length]])
                sub_batches_y.append(sub_batch_y)

            batch_y = np.array(sub_batches_y)
            # shape is currently [batch_size, sequence_length]
            
            # convert batch_y to one-hot
            # using this method: https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e
            batch_y_oh = np.zeros((batch_y.shape[0],batch_y.shape[1], 17))
            
            # this part is [0 ... batch_size, but reshaped into rows
            layer_idx = np.arange(batch_y.shape[0]).reshape(batch_y.shape[0], 1)
            
            # this is [0 ... sequence_length], but batch_size rows
            component_idx = np.tile(np.arange(batch_y.shape[1]), (batch_y.shape[0], 1))
            
            # select indices according to category label
            batch_y_oh[layer_idx, component_idx, batch_y] = 1            
            # batch_y_oh is [batch_size, sequence_length, num_classes]
        
        return batch_x, batch_y_oh

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        l = int(np.ceil( (len(self.combined_utterances) - self.sequence_length + 1) / self.batch_size) )
        if self.algo is "LSTM_CRF":
            return l-1
        else:
            return l

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
