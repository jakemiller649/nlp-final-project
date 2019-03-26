"""Summary of what's contained in this module:

TO DO
- need to finish create vocab function
- need to finish embed matrix function
"""


import os
import xml.etree.ElementTree as ET
import re
from sklearn.model_selection import train_test_split
from collections import Counter
from keras.utils import Sequence
import numpy as np

# import pandas as pd

class Utterance:
    """
    Desription
    Methods
    Attributes
    """
    # len
    # label
    # spit out sentence

    def __init__(self, convo_id):
        self.convo_id = convo_id
        self.da_type = None
        self.length = 0
        self.words = []

    def __len__(self):
        return self.length

class Conversation:
    """
    Desription
    Methods
    Attributes
    """

    def __init__(self, convo_id):
        """
        Args:
        """

        self.convo_id = convo_id  # file name
        self.utterances = []      # list of utterances
        self.labels = []          # list of labels
        self.length = 0           # conversation length
        self.max_utterance_length = 0   # length of longest utterance
        self.vocab = Counter() # unique words and their frequencies

    def add(self, utterance):
        """ Add utterance to this conversation"""

        self.utterances.append(utterance)
        self.length += 1
        self.labels.append(utterance.da_type)
        if len(utterance) > self.max_utterance_length:
            self.max_utterance_length = len(utterance)

    def get_words(self, filedir):
        """adfasdf"""
        from contractions import contractions

        with open(filedir + "\\words\\" + self.convo_id + ".words.xml") as infile:
            # open file, parse xml
            convo_tree = ET.parse(infile)
            convo_root = convo_tree.getroot()

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
    Desription
    Methods
    Attributes
    """

    # create generator function

    def __init__(self, filedir = None, seed = None,
                 max_vocab = 10000, embed_vec = None, embed_dim = 200):
        """Load corpus from file"""

        # initialize some attributes
        self.conversations = []
        self.max_utterance_length = 0
        self.max_conversation_length = 0
        self.utterance_lengths = []
        self.conversation_lengths = []
        self.max_vocab = max_vocab

        if filedir is None:
            filedir = os.getcwd() + "\\data"
        print("Loading corpus from", filedir)

        for file in os.listdir(filedir + "\\dialogueActs"):

            if file.endswith("dialog-act.xml"):
                with open(filedir + "\\dialogueActs\\" + file) as infile:
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
                        # leave da_type as "None"
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
                self.add_conversation(conversation)
        # cue corpus post-processing
        print("Begin corpus post-processing ...")

        # split into test and training
        self.training_split(split_seed = seed)

        # create vocab
        self.create_vocab()

        # create initial embedding matrix
        self.create_embed_matrix(embed_vec, embed_dim)

    def add_conversation(self, conversation):
        self.conversations.append(conversation)

        # keep track of the longest utterances in the corpus
        ### DELETE ###
        if conversation.max_utterance_length > self.max_utterance_length:
            self.max_utterance_length = conversation.max_utterance_length

        # and the longest conversations
        if conversation.length > self.max_conversation_length:
            self.max_conversation_length = conversation.length

    def pad_utterances():
        # figure out 99th percentile of utterance length

        pass
    # pad utterances

    def pad_conversations():
        # figure out 99th percentile of utterance length
        pass
    # pad convos


    # split into train and test (seed)
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

        split_point = floor(len(self.conversations)*0.8)
        shuffle(self.conversations)
        self.train_convos = self.conversations[:split_point]
        self.test_convos = self.conversations[split_point:]

    def create_vocab(self, max_len = None):
        """doc
        Args:
          - max_vocab = number of words you want to include in dictionary (default: 10k)
          - max_len
        """

        print("Creating vocabulary from training set ...")
        self.words = Counter()
        for conversation in self.train_convos:
            self.words += conversation.vocab

        print("Found", len(self.words), "unique words.")
        if len(self.words) > self.max_vocab:
            self.max_vocab = len(self.words)

        ### reusing some code from 266 utils ######
        top_counts = self.words.most_common(self.max_vocab)
        vocab = ["<unk>"] +  [w for w,c in top_counts]

        # maps words to
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}

        # create id sequences for utterances, padding/clipping where appropriate
        for convo in self.train_convos:
            for u in convo.utterances:
                pass
            pass

        for convo in self.test_convos:
            pass



    def create_embed_matrix(self, embed_vec, embed_dim):
        """
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

        docstring
        Args:
          - embed_vec -- one of None (default), 'glove100', 'glove200', 'glove300', 'numberbatch', or 'lexvec'
        """

        print("Building initial embedding matrix ...")

        if embed_vec is None and embed_dim is None:
            raise Exception("If not specifying pretrained vectors, must specify embedding dimension")


        vector_dir = "../../pretrained_vectors/"

        filenames = {"glove100":"glove.6B.100d.txt",
                     "glove200":"glove.6B.200d.txt",
                     "glove300":"glove.6B.300d.txt",
                     "numberbatch":"numberbatch-en.txt", # 300d
                     "lexvec":"lexvec.commoncrawl.300d.W.pos.neg3.vectors"} #300d

        # initialize matrix to zeros
        self.init_embedding_matrix = np.zeros(shape = (self.max_vocab, embed_dim))
        # initialize <unk> token to random
        self.init_embedding_matrix[0] = np.random.uniform(low = -1, high = 1, size = [embed_dim])

        unfound_ids = {i:None for i in range(len(self.words))}
        # open vector file
        with open(vector_dir + filenames[embed_vec], 'r', encoding = 'utf-8') as file:
            for line in file:
                l = line.rstrip("\n").split(" ")
                if l[0] in self.word_to_id:
                    n = self.word_to_id[l[0]]
                    _ = unfound_ids.pop(n, None)
                    # add vector to matrix

        # leave this, make a note of how many words not found
        self.unfound_words = [self.id_to_word[i] for i in unfound_ids]
        #for i in len(self)


class UtteranceGenerator(Sequence):
    """doc string goes here"""

    def __init__(self, corpus, mode, batch_size):
        """
        Args
        Returns
        """
        self.batch_size = batch_size

        if mode == "train":
            convos = corpus.train_convos
        elif mode == "evaluate":
            convos = corpus.test_convos
        else:
            raise Exception("Generator's mode must be either 'train' or 'evaluate'")

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
        batch_x = np.array([u.word_ids for u in self.combined_utterances[start:end]])
        batch_y = np.array([u.da_type for u in self.combined_utterances[start:end]])

        return batch_x, batch_y

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(len(self.combined_utterances) / float(self.batch_size)))
