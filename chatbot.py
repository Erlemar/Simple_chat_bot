import os
import pickle
import numpy as np
import os.path

import nltk
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.optimizers import Adam
from keras.models import Model, model_from_json
import configparser
from keras import backend as K
import re


class ChatBot(object):
    """
    Chatbot.

    The code is based on this repo: https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras
    The idea is the same, but the implementation is different.
    Model is trained using teacher forcing on questions and context.
    """

    def __init__(self, path=''):
        """Define path to files. Other parameters are set automatically using settings.ini."""
        self.path = path
        self.prob = 0
        self.last_query = ' '
        self.text = ' '
        self.set_params()

    def set_params(self):
        """Set parameters and options."""
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(self.path, 'settings.ini'), encoding='utf-8')
        for key, value in self.config['MODEL_PARAMS'].items():
            setattr(self, key, self.config.getint('MODEL_PARAMS', key))

        for key, value in self.config['MAIN'].items():
            setattr(self, key, os.path.join(self.path, value))

    def create_embedding(self):
        """Create embeddings if model is trained from scratch."""
        embeddings_index = {}

        with open(os.path.join(self.glove_dir, self.config['MAIN']['embedding_name'])) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))
        self.embedding_matrix = np.zeros((self.dictionary_size, self.word_embedding_size))

        # Loading vocabulary
        self.vocabulary = pickle.load(open(self.vocabulary_file, 'rb'))
        vocab_indices = [i[0] for i in self.vocabulary]
        # Starting and ending indices.
        self.bos_ind = vocab_indices.index('BOS')
        self.eos_ind = vocab_indices.index('EOS')

        # Using sGlove embedding
        i = 0
        for word in self.vocabulary:
            embedding_vector = embeddings_index.get(word[0])
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            i += 1

    def build_model(self, load_model_type):
        """Model.

        In the original implementation only weights are saved and the model itself is build withing the code.
        If you want to build model from scratch, pass 'weights' to load_model_type and delete weights file from your folder,
        if you want to fine-tune it, pass 'model'.

        """
        def perplexity(y_true, y_pred):
            """Perplexity metric."""
            cross_entropy = K.categorical_crossentropy(y_true, y_pred)
            perplexity = K.pow(2.0, cross_entropy)
            return perplexity

        ad = Adam(lr=0.00005)

        if load_model_type == 'weights':

            input_context = Input(shape=(self.maxlen_input, ), dtype='int32', name='input_context')
            input_answer = Input(shape=(self.maxlen_input, ), dtype='int32', name='input_answer')

            LSTM_encoder = LSTM(self.sentence_embedding_size, kernel_initializer='lecun_uniform')
            LSTM_decoder = LSTM(self.sentence_embedding_size, kernel_initializer='lecun_uniform')

            if os.path.isfile(self.weights_file):
                Shared_Embedding = Embedding(output_dim=self.word_embedding_size,
                                             input_dim=self.dictionary_size,
                                             input_length=self.maxlen_input)
            else:
                Shared_Embedding = Embedding(output_dim=self.word_embedding_size,
                                             input_dim=self.dictionary_size,
                                             weights=[self.embedding_matrix],
                                             input_length=self.maxlen_input)

            word_embedding_context = Shared_Embedding(input_context)
            context_embedding = LSTM_encoder(word_embedding_context)

            word_embedding_answer = Shared_Embedding(input_answer)
            answer_embedding = LSTM_decoder(word_embedding_answer)

            merge_layer = concatenate([context_embedding, answer_embedding])
            out = Dense(int(self.dictionary_size / 2), activation="relu")(merge_layer)
            out = Dense(self.dictionary_size, activation="softmax")(out)

            model = Model(inputs=[input_context, input_answer], outputs=[out])

            model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=[perplexity])

            if os.path.isfile(self.weights_file):
                model.load_weights(self.weights_file)

        elif load_model_type == 'model':
            json_file = open(self.model_json, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(self.model_weights)
            model = loaded_model
            model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=[perplexity])

        self.model = model

    def load_data(self):
        """Load and define data for training."""
        self.questions = pickle.load(open(self.padded_questions_file, 'rb'))
        self.answers = pickle.load(open(self.padded_answers_file, 'rb'))
        self.n = self.answers.shape[0]
        self.step = int(np.around(self.n / self.num_subsets))

    def train_model(self):
        """
        Training the model.

        For each epoch new samples are manually created for teacher forcing.
        Often this could lead to huge requirements of memory, in this case define higher values
        of num_subsets to train on batches.
        """
        for m in range(self.epochs):
            # Loop over training batches due to memory constraints:
            for start_ind in range(0, self.n, self.step):
                cur_question = self.questions[start_ind:start_ind + self.step]
                count = 0
                for i, sent in enumerate(self.answers[start_ind:start_ind + self.step]):
                    l = np.where(sent == self.eos_ind)
                    count += l[0][0] + 1

                Q = np.zeros((count, self.maxlen_input))
                A = np.zeros((count, self.maxlen_input))
                Y = np.zeros((count, self.dictionary_size))

                # Loop over the training examples:
                count = 0
                for i, sent in enumerate(self.answers[start_ind:start_ind + self.step]):
                    ans_partial = np.zeros((1, self.maxlen_input))

                    # Loop over the positions of the current target output (the current output sequence):
                    l = np.where(sent == self.eos_ind)
                    limit = l[0][0]

                    for k in range(1, limit + 1):
                        # Mapping the target output (the next output word) for one-hot codding:
                        y = np.zeros((1, self.dictionary_size))
                        y[0, sent[k]] = 1

                        # preparing the partial answer to input:

                        ans_partial[0, -k:] = sent[0:k]

                        # training the model for one epoch using teacher forcing:

                        Q[count, :] = cur_question[i:i + 1]
                        A[count, :] = ans_partial
                        Y[count, :] = y
                        count += 1

                print('Training epoch: {0}, training examples: {1} - {2}.'.format(m, start_ind, start_ind + self.step))
                self.model.fit([Q, A], Y, batch_size=self.batchsize, epochs=1)

        # Save model.
        model_json_string = self.model.to_json()
        with open(self.model_json, "w") as json_file:
            json_file.write(model_json_string)

        self.model.save_weights(self.model_weights)

    def train(self, load_model_type='weights'):
        """Train model."""
        self.load_data()
        self.create_embedding()
        self.build_model(load_model_type=load_model_type)
        self.train_model()

    def greedy_decoder(self, text):
        """Decode data.

        Flag is a convenience to know whether the text ended or not.
        Probability - probability of answer, used later.
        ans_partial - is an answer which is generated step by step.

        """
        flag = 0
        prob = 1
        ans_partial = np.zeros((1, self.maxlen_input))
        ans_partial[0, -1] = self.bos_ind
        for k in range(self.maxlen_input - 1):
            ye = self.model.predict([text, ans_partial])
            p = np.max(ye[0, :])
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
            if mp == self.eos_ind:
                flag = 1
            if flag == 0:
                prob = prob * p

        # convert indices to text.
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (self.dictionary_size - 2):
                w = self.vocabulary[k]
                text = text + w[0] + ' '
        return(text, prob)

    def preprocess(self, raw_word):
        """Preprocess data."""
        l1 = self.config['LISTS']['to_replace'].split('\n')
        l2 = self.config['LISTS']['replacement'].split('\n')
        l3 = self.config['LISTS']['to_spaces'].split('\n')

        raw_word = raw_word.lower()

        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term, l2[j])

        for term in l3:
            raw_word = raw_word.replace(term, ' ')

        # After replacing there can be situations, when there are no words between dots. Replace with empty string
        # Replacing two or more spaces with empty string.
        raw_word = re.sub('(\s+){2,}', '', raw_word)

        # from original implementation - adding dot, if no punctuation mark at the end of the sentence
        if raw_word.strip().endswith(('!', '?', '.')) == False:
            raw_word = raw_word + ' .'

        # in case there is no text left after cleaning
        if raw_word.strip() in ['!', '?', '.']:
            raw_word = 'what ?'

        return raw_word

    def tokenize(self, sentences):
        """Tokenizing the sentences into words."""
        tokenized_sentences = nltk.word_tokenize(sentences)
        self.vocabulary = pickle.load(open(self.vocabulary_file, 'rb'))
        vocab_indices = [i[0] for i in self.vocabulary]
        self.bos_ind = vocab_indices.index('BOS')
        self.eos_ind = vocab_indices.index('EOS')
        index_to_word = [x[0] for x in self.vocabulary]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        tokenized_sentences = [w if w in word_to_index else self.unknown_token for w in tokenized_sentences]
        X = np.asarray([word_to_index[w] for w in tokenized_sentences])
        s = X.size
        Q = np.zeros((1, self.maxlen_input))
        if s < (self.maxlen_input + 1):
            Q[0, - s:] = X
        else:
            Q[0, :] = X[- self.maxlen_input:]

        return Q

    def start_prediction(self):
        """Load Model."""
        def perplexity(y_true, y_pred):
            cross_entropy = K.categorical_crossentropy(y_true, y_pred)
            perplexity = K.pow(2.0, cross_entropy)
            return perplexity
        json_file = open(self.model_json, 'r')
        ad = Adam(lr=0.00005)
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.model_weights)
        model = loaded_model
        model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=[perplexity])
        self.model = model

    def make_prediction(self, que=''):
        """Generate prediction.

        Take user question as an input, tokenize them and make prediction.
        Last question is saved to use as a context.

        """
        que = self.preprocess(que)
        # Collecting data for further training later:
        q = self.last_query + ' ' + self.text
        a = que
        with open(self.file_saved_context, 'a') as f:
            f.write(q + '\n')
        with open(self.file_saved_answer, 'a') as f:
            f.write(a + '\n')

        # Composing the context:
        if self.prob > 0.2:
            query = self.text + ' ' + que
        else:
            query = que

        # last_text = text
        Q = self.tokenize(query)

        # Using the trained model to predict the answer:
        predout, self.prob = self.greedy_decoder(Q[0:1])
        start_index = predout.find('EOS')

        self.text = self.preprocess(predout[0:start_index])
        # print ('computer: ' + text + '    (with probability of %f)' % prob)
        print(self.text)

        self.last_query = que
        return self.text

    def predict(self, que=''):
        """Predict."""
        # print(que)
        self.start_prediction()
        return self.make_prediction(que)
