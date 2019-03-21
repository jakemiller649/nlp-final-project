"""Summary of what's contained in this module:"""


import os
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

# import numpy as np
# import pandas as pd

class Word:
    """
    Desription
    Methods
    Attributes
    """

    # file
    # id
    # start time,
    # end time, because fuck it just in case
    # duration
    pass

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
        # to add: unique words

    def add(self, utterance):
        """ Add utterance to this conversation"""
        self.utterances.append(utterance)
        self.length += 1
        self.labels.append(utterance.da_type)
        if len(utterance) > self.max_utterance_length:
            self.max_utterance_length = len(utterance)


class Corpus:
    """
    Desription
    Methods
    Attributes
    """

    # read in all convs

    # create generator function

    def __init__(self, filedir = None):
        """Load corpus from file"""

        # initialize some attributes
        self.conversations = []
        self.max_utterance_length = 0
        self.max_conversation_length = 0


        if filedir is None:
            filedir = os.getcwd() + "\\data\\dialogueActs"
        print("Loading corpus from", filedir)

        for file in tqdm_notebook(os.listdir(filedir)):

            if file.endswith("dialog-act.xml"):
                with open(filedir + "\\" + file) as infile:
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

                    # get words from the other file that has the words
                    #utterance.get_words()

                    # add this utterance to our conversation

                    conversation.add(utterance)



                self.add_conversation(conversation)
        # cue corpus post-processing
        print("Begin corpus post-processing")

    def add_conversation(self, conversation):
        self.conversations.append(conversation)

        # keep track of the longest utterances in the corpus
        if conversation.max_utterance_length > self.max_utterance_length:
            self.max_utterance_length = converation.max_utterance_length

        # and the longest conversations
        if conversation.length > self.max_conversation_length:
            self.max_conversation_length = conversation.length

    def pad_utterances():
        pass
    # pad utterances

    def pad_conversations():
        pass
    # pad convos


    # split into train and test (seed)
    def training_split(self, seed = None):
        pass

    def create_vocab(self, max_vocab = 10000):
        pass


    # create vocab for train
    # create pre-trained embedding matrix
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
