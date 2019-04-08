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
import models_gpu as models
import numpy as np
from time import time
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger # maybe more
import json

def run_model(it_no):
    """
    Args:
      - it_no: iteration number just for logging to a file. Originally I used a timestamp but the 
        filenames ended up becoming too long and I was unable to move them around.
    """
  

    BATCH_SIZE = 512
    EPOCHS = 25

    # choose hyperparameters randomly
    d1 = np.random.randint(0,4)
    d2 = np.random.randint(0,4)
    filters = np.random.randint(50,1001)
    kernel_size = np.random.randint(1,11)
    hidden_units = np.random.randint(25,251)
    dropout_rate = round(np.random.uniform(0,1,size=None), 3)
    trainable_embed = np.random.choice([True, False])
    embed_vec = np.random.choice(['glove200', 'glove300', 'numberbatch'])
 


    ## start the timer
    start = time()

    # load the corpus
    corpus = utils.AMI_Corpus(seed = 75, embed_vec = embed_vec)

    cnn = models.CNN(corpus = corpus, d1 = d1, d2 = d2,
                       batch_size = BATCH_SIZE,
                       filters = filters,
                       kernel_size = kernel_size,
                       hidden_units = hidden_units,
                       dropout_rate = dropout_rate,
                       trainable_embed = trainable_embed)

    cnn.model.compile(optimizer = 'adagrad', metrics = ['acc'], loss = 'categorical_crossentropy')

    # create our generators
    ug_train = utils.UtteranceGenerator(corpus, "train", batch_size = BATCH_SIZE, sequence_length = (d1 + d2 + 1), algo = "CNN")
    ug_val = utils.UtteranceGenerator(corpus, "val", batch_size = BATCH_SIZE, sequence_length = (d1 + d2 + 1), algo = "CNN")

    # create keras callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    # log things just in case
    csv_logger = CSVLogger('logs/cnn_history_' + str(it_no) + ".csv") # log epochs in case I want to look back later

    # train the thing
    history = cnn.model.fit_generator(ug_train, epochs=EPOCHS, verbose=1, callbacks=[es, csv_logger],
                                          validation_data=ug_val, #validation_freq=1,
                                          use_multiprocessing=False, shuffle=True)

    # write the results to a csv
    results = [it_no, d1, d2, filters, kernel_size, hidden_units, dropout_rate, embed_vec, history.history['acc'][-5:], 
               history.history['val_acc'][-5:], history.history['loss'][-5:], history.history['val_loss'][-5:], (time() - start), 
               trainable_embed]

    results = ",".join([str(n) for n in results]) + "\n"
    
    with open("logs/cnn_exp_2.csv", "a") as f:
        f.write(results)
    
    print("-----------------------------------------------")

if __name__ == "__main__":
        
    experiment_start = time()
    it_no = 0
    while time() - experiment_start < 7200: # go for two hours, see how it does
        run_model(it_no)
        it_no += 1
