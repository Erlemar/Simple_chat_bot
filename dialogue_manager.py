import os
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances

from chatterbot import ChatBot
from chatbot import ChatBot
from utils import *

import glob
import requests
import re
from datetime import datetime

# these two lines are necessary for Ubuntu. Otherwise it doesn't work.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import twitter
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags


class ThreadRanker(object):
    """
    ThreadRanker class is a class for predicting the most relevant answer to the question related to programming.
    Methods:
        __load_embeddings_by_tag — loads embeddings relevant fot the tag in the question;
        get_best_thread – returns the best answer for the question;
    """

    def __init__(self, config):
        """Loading data."""
        self.word_embeddings, self.embeddings_dim = load_embeddings(config['PATH']['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = config['PATH']['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        """Load embeddings for the single tag to limit memory usage."""
        print(self.thread_embeddings_folder, tag_name)
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """
        Return id of the most similar thread for the question.
        The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        question_vec = np.array(question_to_vec(question, self.word_embeddings, self.embeddings_dim)).reshape(-1, 100)
        
        # initially I used this single line to calculate best_thread, but t2.micro doesn't have enough RAM, so loop is necessary.
        # best_thread = pairwise_distances_argmin(question_vec, thread_embeddings, batch_size=10)[0]
        
        n = int(np.round(thread_embeddings.shape[0] / 10))
        min_value = 1.0
        best_thread = 0
        for i in range(10):
            if i < 8:
                temp_min = np.min(pairwise_distances(question_vec, thread_embeddings[i * n:(i + 1) * n]))
                if temp_min < min_value:
                    min_value = temp_min
                    best_thread = np.argmin(pairwise_distances(question_vec, thread_embeddings[i * n:(i + 1) * n])) + i * n
            else:
                temp_min = np.min(pairwise_distances(question_vec, thread_embeddings[i * n:]))
                if temp_min < min_value:
                    min_value = temp_min
                    best_thread = np.argmin(pairwise_distances(question_vec, thread_embeddings[i * n:])) + i * n

        return thread_ids.values[best_thread]


class DialogueManager(object):
    """
    Class responsible for the main functionality of the bot and for choosing the type of response for user's input.
    Methods:
    create_chitchat_bot - initializes and trains conversational chatbot.
    generate_answer - recognizes intent of the question and gives an appropriate response.
    """
    def __init__(self, config):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(config['PATH']['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(config['PATH']['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\n This thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(config['PATH']['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(config)
        self.chitchat_bot = self.create_chitchat_bot()

        # Twitter keys
        self.twitter_keys = config['TWITTER_API']


    def create_chitchat_bot(self):
        """Initialize self.chitchat_bot with some conversational model."""
        #self.chitchat_bot = ChatBot('Little Bot',
        #                            trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
        #self.chitchat_bot.train("chatterbot.corpus.english")
        self.chitchat_bot = ChatBot()

        return self.chitchat_bot

    def generate_weather_answer(self, question):
        """
        Generate weather forecast for a selected city.
        
        At first try to extract city from user request. If not possible, then generate usual answer.
        Connects to openweathermap and gets forecast, then generates plot of temperature
        in Celsius and Fahrenheit; also shows unique weather conditions.
        """
        # remove previous image, as it isn't needed anymore
        for i in glob.glob(os.path.join(os.getcwd(), '*.png')):
            os.remove(i)

        good_symbols_re = re.compile('[^a-zA-Z -]')
        question_cleaned = good_symbols_re.sub('', question)

        # Extract entities.
        tagged = tree2conlltags(ne_chunk(pos_tag(word_tokenize(question.title()))))
        cities = [i[0] for i in tagged if i[1] == 'NNP']
        city = ''
        for c in cities:
            data = requests.get('http://api.openweathermap.org/data/2.5/forecast?q={0}&appid=f00cf7123615727d162770891d4fd225'.format(c)).json()
            if data['cod'] == '200':
                city = c
                break
        if city == '':
               return self.generate_usual_answer(question)
        else:
            forecast = requests.get('http://api.openweathermap.org/data/2.5/forecast?q={0}&appid=f00cf7123615727d162770891d4fd225'.format(city)).json()
            if forecast['message'] == 'city not found':
                return "I don't know this city!"

            # Generate temperature and date lists for plotting
            date_list = []
            temp_list_c = []
            temp_list_f = []

            for reading in forecast['list']:
                date = datetime.fromtimestamp(int(reading['dt']))
                temperature_c = reading['main']['temp'] - 273.15
                temperature_f = reading['main']['temp'] * 9 / 5 - 459.67
                date_list.append(date)
                temp_list_c.append(temperature_c)
                temp_list_f.append(temperature_f)

            # make chart
            fig, ax = plt.subplots()
            ax.plot_date(date_list, temp_list_c, '-', label='Celsius')
            ax.plot_date(date_list, temp_list_f, '-', label='Fahrenheit')
            ax.grid(True)

            plt.xticks(rotation=30)
            plt.yticks(range(int(min(temp_list_c)) - 1, int(max(temp_list_f) + 1), 5))
            dtFmt = mdates.DateFormatter('%m/%d')
            ax.xaxis.set_major_formatter(dtFmt)
            plt.title('Temperature in {0}'.format(city))
            plt.legend()
            # save image, so it can be sent to user
            plt.savefig('plot.png')

            # List of possible unique weather conditions
            weather = ', '.join(list(set([i['weather'][0]['description'] for i in forecast['list']])))

            return 'Possible weather in the next few days: {0}.;{1}'.format(weather, 'plot.png')

    def generate_twitter_answer(self, question):
        """
        Show the latest tweet for the defined user.
        """
        api = twitter.Api(consumer_key=self.twitter_keys['consumer_key'],
                          consumer_secret=self.twitter_keys['consumer_secret'],
                          access_token_key=self.twitter_keys['access_token_key'],
                          access_token_secret=self.twitter_keys['access_token_secret'])

        good_symbols_re = re.compile('[^a-zA-Z0-9 _]')
        question_cleaned = good_symbols_re.sub('', question.replace('/', ' '))
        account_name = question_cleaned.split(' ')[-1]

        try:
            tweet_id = api.GetUserTimeline(screen_name=account_name, count=1)[0].id
        except twitter.TwitterError:
            return "I don't know this user! Let's continue talking!" + "\n" + self.generate_usual_answer(question)

        return 'https://twitter.com/i/web/status/{0}'.format(tweet_id)

    def generate_usual_answer(self, question):
        """
        Combine stackoverflow and chitchat parts using intent recognition.
        Returns either link to stackoverflow or an answer from chatterbot.
        """

        # Recognize intent of the question using `intent_recognizer`.
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            #response = self.chitchat_bot.get_response(question).text
            response = self.chitchat_bot.predict(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_clasifier to get predictions.
            tag = self.tag_classifier.predict(features)

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag[0])

            return self.ANSWER_TEMPLATE % (tag, thread_id)

    def generate_answer(self, question):
        print('question', question)
        if 'weather' in question.lower():
            return self.generate_weather_answer(question)

        elif 'tweet' in question.lower() or 'twitter' in question.lower():
            return self.generate_twitter_answer(question)

        elif question.lower() == 'today!' or question.lower() == 'today':
            # return a random fact about current date
            return requests.get('http://numbersapi.com/{0}/{1}/date'.format(datetime.today().month, datetime.today().day)).text

        elif question.lower() == 'help!' or question.lower() == 'help':
            return """ This chatbot was created based on the final project of this course, for the honor task:
            https://www.coursera.org/learn/language-processing/home/welcome
            Possible commands:
            sentence with word 'weather' and city name - shows weather forecast using openweathermap api;
            'tweet/twitter account_name' - shows the latest tweet by the user;
            'today!' - shows current date and random fact about it;
            Otherwise bot will either chat or try to answer a programming question.
            Programming languages for which the bot can try to give an answer: c\c++, c#, java, javascript, php, python, r, ruby, swift, vb.
            
            Code on github: https://github.com/Erlemar/Simple_chat_bot
            Chatbot idea is taked from this paper: https://www.researchgate.net/publication/321347271_End-to-end_Adversarial_Learning_for_Generative_Conversational_Agents
            and repo: https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras
            """

        else:
            return self.generate_usual_answer(question)
