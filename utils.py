"""Summary of what's contained in this module:"""


import os
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from collections import Counter

# import numpy as np
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

        with open(filedir + "\\words\\" + self.convo_id + ".words.xml") as infile:
            # open file, parse xml
            convo_tree = ET.parse(infile)
            convo_root = convo_tree.getroot()

        # the leaf numbers do not align perfectly with word numbers, so we need to make sure we are
        # getting the correct words for each utterance
        convo_words = {}
        for leaf in convo_root:
            #if leaf.tag is 'w':
            # going to need some way to handle vocal sounds
            try:
                word_idx = int(leaf.attrib['{http://nite.sourceforge.net/}id'].split('words')[1])
            except:
                # some of them have 'wordsx' and I don't know why
                pass
            w = leaf.text
            convo_words[word_idx] = w

        for utterance in self.utterances:

            # add starting token

            ### JAKE PICK UP HERE #####

            for i in range(utterance.start, utterance.end+1):
                try:
                    word = convo_words[i]
                except:
                    # sometimes they skip numbers
                    pass

                # is it a digit?
                # is it punctuation
                # do pretrained vectors use punctuation???
                # lowercase

                if word is not None:
                    utterance.words.append(word.lower())
                    self.vocab[word] += 1
            # add ending token
            # update token length

class Corpus:
    """
    Desription
    Methods
    Attributes
    """

    # create generator function

    def __init__(self, filedir = None, seed = None, max_vocab = 20000, embed_vec = None):
        """Load corpus from file"""

        # initialize some attributes
        self.conversations = []
        self.max_utterance_length = 0
        self.max_conversation_length = 0
        self.utterance_lengths = []
        self.conversation_lengths = []


        if filedir is None:
            filedir = os.getcwd() + "\\data"
        print("Loading corpus from", filedir)

        for file in tqdm_notebook(os.listdir(filedir + "\\dialogueActs")):

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
        print("Begin corpus post-processing")

        # split into test and training
        self.training_split(split_seed = seed)

        # create vocab
        self.create_vocab(max_vocab)

        # create initial embedding matrix
        create_embed_matrix(self, embed_vec)

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
        print("Splitting corpus into training and test")

        from random import seed, shuffle
        from math import floor

        if split_seed is not None:
            seed(split_seed)

        split_point = floor(len(self.conversations)*0.8)
        shuffle(self.conversations)
        self.train_convos = self.conversations[:split_point]
        self.test_convos = self.conversations[split_point:]

    def create_vocab(self, max_vocab):
        """doc
        Args: max_vocab = number of words you want to include in dictionary (default: 20k)
        """

        print("Creating vocabulary from training set ...")
        self.words = Counter()
        for conversation in self.train_convos:
            self.words += conversation.vocab

        ### JAKE PICK UP HERE ######
        pass

    def words_to_ids():
        """doc"""

        print("Creating word IDs ...")
        # convert words to ids
        # needs to be a default dict that returns <unk> if it's not in there
        pass

    def create_embed_matrix(self, embed_vec):
        """
        docstring
        Args:
          - embed_vec -- one of None (default), 'glove200', 'glove300', 'numberbatch', or 'lexvec'
        """
        vector_dir = "../../pretrained_vectors/"


        # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        # also create unknown tokens -- how to initialize? (randomly with seed?)

        pass
