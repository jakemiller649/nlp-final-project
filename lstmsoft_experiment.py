"""This file tests the CNN model with various hyperparameters.

Hyperparamers to test:

- d1 [0,1,2]
- d2 [0,1,2]
- number of filters [50-1000]
- kernel sizes [1-10]
- dropout_rate [0-1]
- hidden_units of the feed forward layers [25-250]
- embedding layer/dimension
  - I have several pretrained embeddings, including GloVe (100, 200, and 300);
    numberbatch (300), and lexvec (300)
- trainable_embed [True, False] - all the papers I read used pretrained word embeddings
  but none of them said if they let them be trainable during

"""

import utils
import models_tf_only as models
import numpy as np
from time import time
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger # maybe more
import json

def run_model():

    BATCH_SIZE = 512
    EPOCHS = 10 # it seemed to converge by this point anyway

    # choose hyperparameters
    num_layers = np.random.randint(2,16) # lstm layers
    hidden_state_size = np.random.randint(100,400)
    dropout_rate = round(np.random.uniform(0,1,size=None), 3)
    embed_vec = np.random.choice(['glove200', 'glove300', 'numberbatch'])
    bidirectional = np.random.choice([True, False])
    stateful = False
    trainable_embed = np.random.choice([True, False])

    ## start the timer
    start = time()

    # load the corpus
    corpus = utils.AMI_Corpus(seed = 75, embed_vec = embed_vec)

    lstm = models.LSTMSoftmax(corpus,
                              batch_size = BATCH_SIZE,
                              num_layers = num_layers,
                              dropout_rate = dropout_rate,
                              hidden_state_size = hidden_state_size,
                              stateful = stateful, bidirectional = bidirectional, trainable_embed = trainable_embed)

    lstm.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')

    # create our generators
    ug_train = utils.UtteranceGenerator(corpus, "train", batch_size = BATCH_SIZE)
    ug_val = utils.UtteranceGenerator(corpus, "val", batch_size = BATCH_SIZE)

    # create keras callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    right_now = datetime.now().isoformat() # timestamp
    csv_logger = CSVLogger('logs/lstm_history_' + right_now) # log epochs in case I want to look back later

    # note to self, maybe change validation_steps and validation_freq
    history = lstm.model.fit_generator(ug_train, epochs=EPOCHS, verbose=1, callbacks=[es, csv_logger],
                      validation_data=ug_val, #validation_steps=100, #validation_freq=1,
                      use_multiprocessing=False, shuffle=True)

    results = {"num_layers":num_layers, "hidden_state_size":hidden_state_size,"dropout_rate":dropout_rate,
               "embed_vec":embed_vec, "acc":history.history['acc'][-5:], "val_acc":history.history['acc'][-5:],
               "loss":history.history['loss'][-5:], "val_loss":history.history['val_loss'][-5:], "time":time() - start,
                "trainable_embed":trainable_embed, "stateful":stateful, "bidirectional":bidirectional}
    
    results = {str(k):str(v) for k,v in results.items()}

    with open("logs/lstm_params_" + right_now + ".json", "w") as f:
        json.dump(results, f)
        
    print("-------------------")


if __name__ == "__main__":
    experiment_start = time()
    while time() - experiment_start < 7200: # start with two hours, see how it does
        run_model()
