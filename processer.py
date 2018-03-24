# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from collections import Counter
from keras.preprocessing import sequence
import configparser


class DataProcesser(object):
    """
    Class for processing data.

    At first it is splited into context and answers.
    Texts are tokenized, vocabulary with words and their ids is created if necessary.
    Texts are encoded using it.
    """

    def __init__(self, path='',
                 separator='==='):
        """Init."""
        params = {'path': path,
                  # 'vocabulary_size': vocabulary_size,
                  # 'maxlen_input': maxlen_input,
                  # 'maxlen_output': maxlen_output,
                  'separator': separator}

        self.set_params(**params)

    def set_params(self, **params):
        """Set parameters and options."""
        for key, value in params.items():
            setattr(self, key, value)

        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(self.path, 'settings.ini'))
        for key, value in self.config['MAIN'].items():
            setattr(self, key, os.path.join(self.path, value))

        self.dictionary_size = self.config.getint('MODEL_PARAMS', 'dictionary_size')
        self.maxlen_input = self.config.getint('MODEL_PARAMS', 'maxlen_input')
        self.maxlen_output = self.config.getint('MODEL_PARAMS', 'maxlen_output')

    def save_file(self, data, file_name):
        """Save file to pickle object."""
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def split_text(self):
        """Clean text and split it into context and answers.

        At first text is cleaned: words with apostrophe are manually replaced with full versions.
        Spaces are added to punctuation. Symbols denoting author of the replique are replaced with spaces.
        Context if the current question + previous answer. If previous answer if empty (dialogue is only starting),
        data isn't saved.
        """
        self.questions = []
        self.answers = []
        pre_previous_raw = ''
        previous_raw = ''

        l1 = self.config['LISTS']['to_replace'].split('\n')
        l2 = self.config['LISTS']['replacement'].split('\n')
        l3 = self.config['LISTS']['to_spaces'].split('\n')

        for i, raw_word in enumerate(self.text):
            if self.separator in raw_word:
                # if new dialogue is started, clear previous utterances
                pre_previous_raw = ''
                previous_raw = ''
            else:
                for j, term in enumerate(l1):
                    raw_word = raw_word.replace(term, l2[j])
                for term in l3:
                    raw_word = raw_word.replace(term, ' ')

                raw_word = raw_word.lower()

                if previous_raw != '' and pre_previous_raw != '':
                    self.questions.append(pre_previous_raw[:-1] + ' ' + previous_raw[:-1])
                    self.answers.append(raw_word[:-1])

                pre_previous_raw = previous_raw
                previous_raw = raw_word

        self.all = self.answers + self.questions

    def read_data(self, files=[]):
        """Read data."""
        text = []
        print(files)
        for arg in files:
            with open(arg, 'r', encoding='utf-8') as f:
                text.append(f.readlines())
        if len(files) > 1:
            text = [i for j in text for i in j]
        else:
            # if only 1 file with text
            text = text[0]
        self.text = text

    def tokenize(self):
        """Tokenize data."""
        print("Tokenizing the answers...")
        paragraphs_a = ['BOS ' + p + ' EOS' for p in self.answers]
        self.tokenized_answers = [p.split() for p in paragraphs_a]

        paragraphs_q = [p for p in self.questions]
        self.tokenized_questions = [p.split() for p in paragraphs_q]

        paragraphs_b = ['BOS ' + p + ' EOS' for p in self.all]
        self.tokenized_text = ' '.join(paragraphs_b).split()

    def calculate_freq(self):
        """Calculate word frequencies and create vocab."""
        # Counting the word frequencies:
        self.word_freq = Counter(self.tokenized_text)
        print("Found {0} unique words tokens.".format(len(self.word_freq)))

        # Get the most common words
        self.vocab = self.word_freq.most_common(self.dictionary_size - 1)

        # Saving vocabulary:
        self.save_file(self.vocab, self.dictionary_size)

    def encode(self):
        """Build index_to_word and encode texts."""
        index_to_word = [x[0] for x in self.vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = {w: i for i, w in enumerate(index_to_word)}
        print("Using vocabulary of size {0}.".format(self.dictionary_size))
        print("The least frequent word in our vocabulary is {0} and appeared {1} times.".format(self.vocab[-1][0],
                                                                                                self.vocab[-1][1]))

        # Replacing all words not in our vocabulary with the unknown token:
        self.tokenized_answers = [[w if w in word_to_index else self.unknown_token for w in sent] for sent in self.tokenized_answers]
        self.tokenized_questions = [[w if w in word_to_index else self.unknown_token for w in sent] for sent in self.tokenized_questions]

        # Creating training data
        X = np.asarray([[word_to_index[w] for w in sent] for sent in self.tokenized_questions])
        Y = np.asarray([[word_to_index[w] for w in sent] for sent in self.tokenized_answers])

        Q = sequence.pad_sequences(X, maxlen=self.maxlen_input)
        A = sequence.pad_sequences(Y, maxlen=self.maxlen_output, padding='post')

        self.save_file(Q, self.padded_questions_file)
        self.save_file(A, self.padded_answers_file)

    def process_text_initial(self, *argv):
        """Use this when you want to train model from scratch. In this case new vocabulary is created.
        Pass list of files to the method."""
        self.read_data(*argv)
        self.split_text()
        self.tokenize()
        self.calculate_freq()
        self.encode()

    def add_data(self, files=[]):
        """Use this when you want to train model on additional data. In this case existing vocabulary is used.
        Pass list of files to the method."""
        self.read_data(files)
        self.split_text()
        self.tokenize()
        with open(self.vocabulary_file, 'rb') as f:
            self.vocab = pickle.load(f)
        self.encode()
