from importlib import reload
import utils
import models #_gpu as models
import numpy as np
from time import time
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger # maybe more
import json


def run_kumar_model(it_no):


    BATCH_SIZE = 252
    EPOCHS = 10 # it seemed to converge by this point anyway

    # choose hyperparameters
    num_layers = np.random.randint(1,5) # lstm layers
    hidden_state_size = np.random.randint(100,400)
    dropout_rate = round(np.random.uniform(0,1,size=None), 3)
    #embed_vec = np.random.choice(['glove200', 'glove300', 'numberbatch'])
    bidirectional = True
    stateful = False
    trainable_embed = True
    
    #num_layers = 1
    #hidden_state_size = 300
    #dropout_rate = 0.2
    embed_vec = 'glove300'
    sequence_length = 7

    ## start the timer
    start = time()

    # load the corpus
    corpus = utils.AMI_Corpus(seed = 75, embed_vec = embed_vec)

    bi_lstm = models.BiLSTMCRF(corpus,
                               batch_size = BATCH_SIZE,
                               sequence_length = sequence_length,
                               num_layers = num_layers,
                               dropout_rate = dropout_rate,
                               hidden_state_size = hidden_state_size,
                               stateful = stateful, bidirectional = bidirectional, 
                               trainable_embed = trainable_embed)

    bi_lstm.compile()

    ug_train = utils.UtteranceGenerator(corpus, "train", batch_size = BATCH_SIZE, sequence_length = sequence_length, algo = "LSTM_CRF")
    ug_val = utils.UtteranceGenerator(corpus, "val", batch_size = BATCH_SIZE, sequence_length = sequence_length, algo = "LSTM_CRF")

    # create keras callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    #right_now = datetime.now().isoformat() # timestamp
    csv_logger = CSVLogger('logs/kumar_history_' + str(it_no) + ".csv") # log epochs in case I want to look back later

    # note to self, maybe change validation_steps and validation_freq
    history = bi_lstm.model.fit_generator(ug_train, epochs=EPOCHS, verbose=1, callbacks=[es, csv_logger],
                                          validation_data=ug_val)
  
    # write the results to a csv
    results = [it_no, num_layers, hidden_state_size, dropout_rate, embed_vec, history.history['crf_viterbi_accuracy'][-5:], 
               history.history['val_crf_viterbi_accuracy'][-5:], history.history['loss'][-5:], 
               history.history['val_loss'][-5:], (time() - start), trainable_embed, stateful, bidirectional]

    results = ",".join([str(n) for n in results]) + "\n"
    
    with open("logs/lstmcrf_exp_2.csv", "a") as f:
        f.write(results)
    
    print("-------------------")

if __name__ == "__main__":
        
    experiment_start = time()
    
    with open("logs/lstmcrf_exp_2.csv", "w") as f:
        headers = """it_no, num_layers, hidden_state_size, dropout_rate, embed_vec, acc, val_acc, loss, val_loss, time, trainable_embed, stateful, bidirectional\n"""
        f.write(headers)
    
    it_no = 10
    while time() - experiment_start < 3600: # go for one hour
        run_kumar_model(it_no)
        it_no += 1