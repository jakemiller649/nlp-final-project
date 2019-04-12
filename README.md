# W266 Final Project
This repo contains the code from my final project for UC-Berkeley MIDS Program W266, Natural Language Processing with Deep Learning.

A few useful pointers:
- models.py. This module contains the code for the LSTM-Softmax and CNN models from my paper
- models_crf.py. This module contains the LSTM-CRF module. (I pulled it into a separate file because the CRF layer from keras_contrib only works with Keras layers, not tf.keras layers, so this file is all regular keras.)
- utils.py. This contains the functions I wrote to extract the corpus from raw XML files, build a vocabulary, and create an initial embedding matrix. It also contains a batch generator class (based on tf.keras' Sequence) that produced different batches depending on the model being tested.
- cnn_experiment.py, lstmsoft_experiment.py, and BiLSTMCRF_experiment.py. These were just files I used to run experiments. I called them from run_trials.sh, which ran experiments, logged results, then shut down my instance, so I could have all this going in the background while doing other stuff without worrying about my instance remaining on unnecessarily. 
