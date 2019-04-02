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

jake, don't forget to write some sort of logging function

"""

import utils
import models
import numpy as np
from time import time
from datetime import datetime
from keras.callbacks import EarlyStopping, CSVLogger # maybe more
import json

def generate_random_params():
    """Generates random hyperparameters from predefined ranges or sets"""

    # choose hyperparameters
    BATCH_SIZE = 25
    d1 = np.random.randint(0,3)
    d2 = np.random.randint(0,3)
    filters = np.random.randint(50,1001)
    kernel_size = np.random.randint(1,11)
    hidden_units = np.random.randint(25,251)
    dropout_rate = round(np.random.uniform(0,1,size=None), 3)
    embed_vec = np.random.choice(['glove100', 'glove200', 'glove300', 'numberbatch', 'lexvec'])

    ## start the timer
    start = time()

    # load the corpus
    corpus = utils.AMI_Corpus(seed = 75, embed_vec = embed_vec)

    cnn = models.CNN(corpus = corpus,
                       batch_size = BATCH_SIZE,
                       filters = filters,
                       kernel_size = kernel_size,
                       hidden_units = hidden_units,
                       dropout_rate = dropout_rate)

    cnn.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')

    # create our generator
    ug = utils.UtteranceGenerator(corpus, "train", batch_size = BATCH_SIZE, sequence_length = (d1 + d2 + 1))

    # create keras callbacks
    es = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0)

    right_now = datetime.now().isoformat() # timestamp
    csv_logger = CSVLogger('logs/history_' + right_now) # log epochs in case I want to look back later

    history = model.fit_generator(ug, epochs=50, verbose=1,
                      callbacks=None,
                      validation_data=None, validation_steps=None, validation_freq=1,
                      use_multiprocessing=True, shuffle=True)

    results = {"d1":d1, "d2":d2, "filters":filters, "kernel_size":kernel_size, "hidden_units":hidden_units, "dropout_rate":dropout_rate,
               "embed_vec":embed_vec, "acc":history.history['acc'][-5:], "val_acc":history.history['acc'][-5:],
               "loss":history.history['loss'][-5:], "val_loss":history.history['val_loss'][-5:], time:time() - start}

    with open("logs/params_" + right_now + ".json", "w") as f:
        json.dump(results, f)
