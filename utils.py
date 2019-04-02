"""Summary of what's contained in this module:

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
        self.vocab = Counter() # unique words and their frequencies

    def add(self, utterance):
        """ Add utterance to this conversation"""

        self.utterances.append(utterance)
        self.length += 1
        self.labels.append(utterance.da_type)

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

    def __init__(self, filedir = None, seed = None,
                 max_vocab = 10000, max_utt_length = None, max_convo_len = None,
                 embed_vec = None, embed_dim = 100):
        """Load corpus from file"""

        # initialize some attributes
        self.conversations = []
        self.utterance_lengths = []
        self.conversation_lengths = []
        self.max_vocab = max_vocab
        self.max_utt_length = max_utt_length
        self.max_convo_len = max_convo_len

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
                self.conversations.append(conversation)

        # cue corpus post-processing
        print("Begin corpus post-processing ...")

        # split into test and training
        self.training_split(split_seed = seed)

        # create vocab
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
        """doc
        Args:
          - max_vocab = number of words you want to include in dictionary (default: 10k)
          - max_utt_length len = maximum utterance length; calculated below if None provided
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
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

        docstring
        Args:
          - embed_vec -- one of None (default),
            'glove100', 'glove200', 'glove300', 'numberbatch', or 'lexvec'
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

            # leave this, make a note of how many words not found
            #self.unfound_words = [self.id_to_word[i] for i in unfound_ids]
            self.embed_dim = embed_dim
            self.unfound_words = len(unfound_ids)
            #for i in len(self)

        else:
            # no pretrained vectors, just initialize a random matrix
            self.init_embedding_matrix = np.random.uniform(low = -1, high = 1, size = [self.max_vocab + 1, embed_dim])
            print(self.init_embedding_matrix.shape)
            self.embed_dim = embed_dim
            self.unfound_words = None

    def pad_clip(convos, max_convo_len):
        """Function for padding/clipping conversations (next function explains why this
        may occur."""

        for c in convos:
            if c.length == max_len:
                # just right, good to go
                continue
            elif c.length > max_len:
                # conversation is too long
                c.utterances = c.utterances[:max_len]
                c.length = max_len
            else:
                # conversation is too short
                how_short = max_len - c.length
                for _ in how_short:
                    dummy_utterance = Utterance(convo_id = c.convo_id)
                    # <pad> is 0 in our ID table, so we can just use np.zeros
                    dummy_utterance.word_ids = np.zeros(max_utt_length)
                    # I might have to fix this part??
                    dummy_utterance.da_type = None
                    c.add(dummy_utterance)


    def pad_convos(self):
        """While two of the models I am testing take either one utterance at a time as input, or a short
        sequence of utterances, Kumar et al 2018 takes whole conversations as inputs. So padding/clipping
        here as appropriate. (This function is only called within )"""

        if self.max_convo_len is None:
            c_lens = [c.length for c in self.train_convos]
            self.max_convo_len = floor(np.percentile(c_lens, 95))


class UtteranceGenerator(Sequence):
    """
    Inherits Keras Sequence class to optimize multiprocessing
    doc string goes here"""

    def __init__(self, corpus, mode, batch_size, sequence_length = 1):
        """
        Args
        Returns
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        assert self.sequence_length >= 1

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

        if self.sequence_length == 1:
            batch_x = np.array([u.word_ids for u in self.combined_utterances[start:end]])
            batch_y = np.array([u.da_type for u in self.combined_utterances[start:end]])
        else:
            sub_batches_x = []
            sub_batches_y = []
            for idx in range(start,end):
                sub_batch_x = [u.word_ids for u in self.combined_utterances[(idx - self.sequence_length + 1):idx+1]]
                sub_batch_y = [u.da_type for u in self.combined_utterances[(idx - self.sequence_length + 1):idx+1]]
                sub_batches_x.append(sub_batch_x)
                sub_batches_y.append(sub_batch_y)

            batch_x = np.array(sub_batches_x)
            batch_y = np.array(sub_batches_y)
            # batch x shape is [batch_size, sequence_length, utterance_length]
            # batch y shape is [batch_size, sequence_length]

        return batch_x, batch_y

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(len(self.combined_utterances) / float(self.batch_size)))

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(self.sequence_length - 1, len(self))):
            # so if sequence length is 1, you start at idx 0. if it's 5 you start at idx 4
            yield item
