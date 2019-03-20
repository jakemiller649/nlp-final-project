https://www.nltk.org/api/nltk.html
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

notes on data generators: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
and: https://keras.io/utils/#sequence

notes on CRF usage: https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/


to figure out:
- how to
- did goo and chen use attention?
- did any of them use attention? how does one use attention?
- take all the hyperparameters and do a linear regression to show impact of each one (??)
- BERT???

what is attention:
-


to do:
- benchmark with Naive Bayes/random forest,  tf-idf bigrams as features
- might need to evaluate as conversations (to do Kumar) or as individual sentences
- may need to do some coding on original to make this work
- what do Goo do? As convos or as sentences?
- then it becomes -- do I split on conversation, or on sentence (is there a equivalent to clustering??)

- I guess I do a basic version with each utterance as input ... then a fancier version with conversations as inputs?

common elements:
- embedding layer

so I need the following structure
- conversation
  - sentence
    - word / punctuation / EOS

embeddings
- this looks like something I should probably know for JL too: https://github.com/RaRe-Technologies/gensim
- alternative embedding: https://github.com/alexandres/lexvec
- glove download: https://nlp.stanford.edu/projects/glove/
- sense2vec?

lee and dernoncourt https://arxiv.org/pdf/1603.03827.pdf in short, it manufactures features from each sentence
- convolutional layer takes x_l words as inputs
 - applies some number of convolutions, then pooling, then outputs a vector s (S2V, short text to vector)
 - so each s2v could feed into its own feed foward layer, or into one feed foward layer, depending on the "history" chosen
 - each s2v comes from each "short text" ... you have the option to feed in the previous
 - hyper parameters d1 and d2 are history sizes used in each layer
 - 10 epochs? also what is early stopping, I forget
   - early stopping means you stop training when no progress is made on validation after x many epochs. x here is called the "patience"
 - also, first FF layer is tanh, second is softmax
 - d from 0 to 1 was chosen
 - pretrained word embeddings on glove (200, twitter) or word2vec (300, google news)

kumar: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16706/16724
- embedding layer
- utterance layer that is a bidirectional LSTM
- pooling layer
- another bi-LSTM layer that represents the converation (can I do this??)
- https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn

- early stopping with a patience of 5 epochs?

khanpour: http://www.aclweb.org/anthology/C16-1189
- they tried a lot of difference w2v and GloVe options
- ltsm layers -- between 2 and 15, 5 and 10 at the best
- output layer is not specified, is it softmax?

PER, LOC, ORG, MISC
