"""
This module contains various functions and classes that help take the dialog acts from the raw XML files (there are XML files that 
contain the conversations themselves, plus XML files that contain the DAs) to something that can be fed into a neural network. 

My basic hierarchy of classes is:
AMI_Corpus
|--Conversation
   |--Utterance

This file also includes UtteranceGenerator, which is a batch iterator that inherits keras's Sequence class, which is optimized
for batch processing.

"""


import os
import xml.etree.ElementTree as ET
import re
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.utils import Sequence
import numpy as np


class Utterance:
    """
    Each utterance has words, length, a da_type (label) and a convseration (convo_id)
    that it is associated with.
    """

    def __init__(self, convo_id):
        self.convo_id = convo_id
        self.da_type = None
        self.length = 0
        self.words = []

    def __len__(self):
        return self.length

class Conversation:
    """
    Desription: Subdivision of corpus. Consists of utterances
    Methods: initialize, add utterance, get words from 
    Attributes: convo_id, a list of utterances, a list of corresponding labels, a length, and a vocabulary
    """

    def __init__(self, convo_id):
        """
        Args: conversation id
        """

        self.convo_id = convo_id  # file name
        self.utterances = []      # list of utterances
        self.labels = []          # list of labels
        self.length = 0           # conversation length
        self.vocab = Counter() # unique words and their frequencies

    def add(self, utterance):
        """ Add utterance to this conversation"""

        self.utterances.append(utterance)
        self.length += 1
        self.labels.append(utterance.da_type)

    def get_words(self, filedir):
        """The DA files (one per conversation) just have a list of utterances, their DA type, and word indices from a different 
        conversation file. This function goes to the second file, pulls out the words, and adds them to a global vocabulary """
        
        # side note, I am handling contractions by getting rid of them
        from contractions import contractions

        
        with open(filedir + "/words/" + self.convo_id + ".words.xml") as infile:
            # open file, parse xml
            convo_tree = ET.parse(infile)
            convo_root = convo_tree.getroot()

        # pull out all the words from the words file, with their word index
        # the leaf numbers do not align perfectly with word numbers, so we need to make sure we are
        # getting the correct words for each utterance
        convo_words = {}
        for leaf in convo_root:
            try:
                word_idx = int(leaf.attrib['{http://nite.sourceforge.net/}id'].split('words')[1])
            except:
                # some of them have 'wordsx' and I don't know why
                # the ones that do don't seem to have words
                pass
            w = leaf.text
            convo_words[word_idx] = w

        # go back to the utterances, see what word indices they correspond to, and fill them in
        for utterance in self.utterances:
            for i in range(utterance.start, utterance.end+1):

                # this to track whether the word should split into multiple words
                multiple = False

                try:
                    # sometimes they skip numbers, only a couple outlier cases
                    word = convo_words[i]
                except:
                    continue

                if word is None:
                    # sometimes this is a thing, there is no word
                    continue

                # lowercase
                word = word.lower()

                # is it punctuation
                if word in [".", "?"]:
                    # replace with end of sentence tag
                    word = "<eos>"
                elif word is ",":
                    # skip, we don't really need commas
                    continue

                # contraction
                if "\'" in word:
                    # either return the spelled out version, or if it's
                    # not in there just return the original word
                    # and it will not get a pretrained vector
                    try:
                        word = contractions[word]
                        multiple = True
                    except:
                        # leave word as is
                        pass

                # a couple more ...
                # <name>'ll generally not caught -> <name> will
                if "\'ll" in word:
                    word = re.sub(r"'ll", " will", word)
                    multiple = True

                if "\'s" in word:
                    # treat 's as its own token
                    word = re.sub(r"'s", r" 's", word)
                    multiple = True

                # acronyms are generally handled f_b_i_
                if "_" in word:
                    # just smush together -> fbi
                    word = re.sub(r"_", "", word)

                # split up hyphenated word
                if "-" in word:
                    word = re.sub(r"-", " ", word)
                    multiple = True

                if multiple is False:
                    utterance.words.append(word)
                    self.vocab[word] += 1
                else:
                    # if our word is now multiple words
                    for w in word.split(" "):
                        utterance.words.append(w)
                        self.vocab[w] += 1


class AMI_Corpus:
    """
    Most important class, it loads the corpus from the files, splits it, creates vocab, and much more!
    
    Methods: initialize, test/train split, create vocab, create embedding matrix
    
    """

    def __init__(self, filedir = None, seed = None,
                 max_vocab = 10000, max_utt_length = None,
                 embed_vec = None, embed_dim = 100):
        """
        Args:
         - filedir: Where are the XML files. If none, defaults to file structure provided by the AMI Corpus
           download.
         - seed. Random seed for training/split, which I used for reproducibility
         - max_vocab. Defaults to 10,000 but it turns out there aren't that many words in there
         - max_utt_length: max utterance length, if you want to pad/clip at a certain length. Otherwise it uses 
           99th percentile length
         - embed_vec: choose from several pretrained word embeddings options
         - embed_dim: if you don't choose pretrained, you must choose an embedding dimension
        """

        # initialize some attributes
        self.conversations = []
        self.utterance_lengths = []
        self.conversation_lengths = []
        self.max_vocab = max_vocab
        self.max_utt_length = max_utt_length

        if filedir is None:
            filedir = os.getcwd() + "/data"
        print("Loading corpus from", filedir)

        for file in os.listdir(filedir + "/dialogueActs"):

            if file.endswith("dialog-act.xml"):
                with open(filedir + "/dialogueActs/" + file) as infile:
                    # open file, parse xml
                    convo_tree = ET.parse(infile)
                    convo_root = convo_tree.getroot()

                # initialize Conversation object
                convo_id = file.replace(".dialog-act.xml", "")
                conversation = Conversation(convo_id)

                for leaf in convo_root:

                    # initialize utterance object
                    utterance = Utterance(convo_id = convo_id)

                    if len(leaf) == 2:
                        # not all leaves have DA types attached, and the ones that do
                        # it's always in the second position
                        utterance.da_type = int(re.findall(r'[0-9]{1,2}', leaf[0].attrib['href'])[-1])
                        b = 1
                    else:
                        # leave da_type as 0
                        utterance.da_type = 0
                        b = 0
                    # note start and end points
                    if ".." in leaf[b].attrib['href']:
                        # the ellipsis above indicates that the utterance is more than one word
                        start, end = leaf[b].attrib['href'].split("..")
                        s = re.search(r'words[0-9]{1,4}', start)
                        e = re.search(r'words[0-9]{1,4}', end)


                        utterance.start = int(s[0][5:])
                        utterance.end = int(e[0][5:])
                        utterance.length = 1 + int(e[0][5:]) - int(s[0][5:])

                    else:
                        start = leaf[b].attrib['href']
                        s = re.search(r'words[0-9]{1,4}', start)

                        # start and end are the same
                        utterance.start = int(s[0][5:])
                        utterance.end = int(s[0][5:])
                        utterance.length = 1

                    # add this utterance to our conversation
                    self.utterance_lengths.append(utterance.length)
                    conversation.add(utterance)

                # get words from the other file that has the words
                w = conversation.get_words(filedir = filedir)

                # add converation to corpus
                self.conversation_lengths.append(conversation.length)
                self.conversations.append(conversation)

        # cue corpus post-processing
        print("Begin corpus post-processing ...")

        # split into test and training
        self.training_split(split_seed = seed)

        # create vocab on training set
        self.create_vocab()

        # create initial embedding matrix
        self.create_embed_matrix(embed_vec, embed_dim)

    def training_split(self, split_seed = None):
        """ Splits corpus into training and test sets. Split is done at the conversation level
        Args: split_seed (a random seed for reproducibility)
        Returns: training conversations, test conversations
        """
        print("Splitting corpus into training and test ...")

        from random import seed, shuffle
        from math import floor

        if split_seed is not None:
            seed(split_seed)

        split_point_1 = floor(len(self.conversations)*0.7)
        split_point_2 = floor(len(self.conversations)*0.8)

        shuffle(self.conversations)
        self.train_convos = self.conversations[:split_point_1]
        self.val_convos = self.conversations[split_point_1:split_point_2]
        self.test_convos = self.conversations[split_point_2:]

    def create_vocab(self):
        """
        Creates a vocabulary with word_to_id and id_to_word dicts. Some lines of code lifted from the 
        w266 utils file. 
        
        Note that this is called after training/test split -- it only builds vocab and embed matrix 
        on words from training set
        
        """
        from math import floor

        print("Creating vocabulary from training set ...")
        self.words = Counter()
        for conversation in self.train_convos:
            self.words += conversation.vocab

        print("Found", len(self.words), "unique words.")
        if len(self.words) < self.max_vocab:
            self.max_vocab = len(self.words)

        ### reusing some code from 266 utils ######
        top_counts = self.words.most_common(self.max_vocab)
        vocab = ["<pad>", "<unk>"] +  [w for w,c in top_counts]

        # maps words to
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}

        # create id sequences for utterances, padding/clipping where appropriate
        if self.max_utt_length is None:
            self.max_utt_length = floor(np.percentile(self.utterance_lengths, 99))

        def pad(convos):
            for convo in convos:
                for u in convo.utterances:
                    # 1 maps to <unk>
                    ids = [self.word_to_id.get(i, 1) for i in u.words]
                    if len(ids) >= self.max_utt_length:
                        # clip
                        u.word_ids = np.array(ids[:self.max_utt_length])
                    else:
                        # pad
                        ids = ids + (self.max_utt_length - len(ids))*[0]
                        u.word_ids = np.array(ids)

        pad(self.train_convos)
        pad(self.val_convos)
        pad(self.test_convos)

    def create_embed_matrix(self, embed_vec, embed_dim):
        """
        This function adapts the following blog post for using pretrained word embeddings:
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

        Args:
          - embed_vec -- one of None (default),
            'glove100', 'glove200', 'glove300', 'numberbatch', or 'lexvec'
          - embed_dim, which you must provide if not using pretrained vectors
        """

        print("Building initial embedding matrix ...")

        if embed_vec is None and embed_dim is None:
            raise Exception("If not specifying pretrained vectors, must specify embedding dimension")


        vector_dir = "../../pretrained_vectors/"

        filenames = {"glove100":["glove.6B.100d.txt", 100],
                     "glove200":["glove.6B.200d.txt", 200],
                     "glove300":["glove.6B.300d.txt", 300],
                     "numberbatch":["numberbatch-en.txt", 300], # 300d
                     "lexvec":["lexvec.commoncrawl.300d.W.pos.neg3.vectors", 300]} #300d

        if embed_vec is not None:
            embed_dim = filenames[embed_vec][1]

            # initialize matrix to random
            self.init_embedding_matrix = np.random.uniform(low = -1, high = 1, size = [self.max_vocab + 2, embed_dim])
            print(self.init_embedding_matrix.shape)

            unfound_ids = {i:None for i in range(len(self.words))}
            # open vector file
            print("loading pretrained vectors from", filenames[embed_vec][0])
            with open(vector_dir + filenames[embed_vec][0], 'r', encoding = 'utf-8') as file:
                for line in file:
                    l = line.rstrip("\n").split(" ")
                    if l[0] in self.word_to_id:
                        # mark it as found for internal tracking
                        n = self.word_to_id[l[0]]
                        _ = unfound_ids.pop(n, None)

                        # add vector to matrix
                        self.init_embedding_matrix[n] = [float(i) for i in l[1:]]

            # note to self, how many unfound words are there
            self.embed_dim = embed_dim
            self.unfound_words = len(unfound_ids)

        else:
            # no pretrained vectors, just initialize a random matrix
            self.init_embedding_matrix = np.random.uniform(low = -1, high = 1, size = [self.max_vocab + 1, embed_dim])
            print(self.init_embedding_matrix.shape)
            self.embed_dim = embed_dim
            self.unfound_words = None


class UtteranceGenerator(Sequence):
    """
    Batch generator based on Keras's Sequence class. Generates either one utterance per batch, or a sequence
    of utterances.
    """

    def __init__(self, corpus, mode, batch_size, sequence_length = 1, algo = None):
        """
        Args
        - corpus: must be an AMI_Corpus that has already done training/test split and create vocab steps
        - mode: training, test, or validation
        - sequence_length: defaults to 1
        - algo: Must be one of LSTM_Soft, CNN, or LSTM_CRF. Why, you ask?
        
        Each of the three algorithms tested in this project takes slightly different combinations of x/y:
        - LSTM_Soft generates one utterance for x and one label for y
        - CNN generates a sequence of utterances for x but one label for y (that corresponds to the last utterance)
        - LSTM_CRF generates a sequence of utterances for x and a corresponding sequence of labels for y for using in the CRF
          output layer
        
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
