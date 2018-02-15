import pickle
import json
import re
import numpy as np
from nltk.corpus import stopwords
import configparser



def text_prepare(text):
    """Perform tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Load pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    config = configparser.ConfigParser()
    config.read('settings.ini')
    with open(config['PATH']['WORD_EMBEDDINGS'], 'r') as f:
        embeddings = json.load(f)
    return embeddings, len(embeddings['using'])


def question_to_vec(question, embeddings, dim):
    """Transform a string to an embedding by averaging word embeddings."""
    result = [embeddings[w] for w in question.split() if w in embeddings]
    return np.mean(result, axis=0) if len(result) > 0 else np.zeros((dim, ))


def unpickle_file(filename):
    """Return the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
