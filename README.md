## Basic NLP chat-bot

This chatbot was created based on the final project of this course: https://www.coursera.org/learn/language-processing/home/welcome and later updated to meet the requirements of the honor assignment.

The main functionality of the bot is to distinguish two types of questions (questions related to programming and others) and then either give an answer or talk using a conversational model.

### Distinguishing intent

At first I had two datasets: StackOverflow posts (programming questions for 10 languages) and dialogue phrases from movie subtitles (non-programming questions). TfidfVectorizer with LogisticRegression were used to build model to classify user's question into these categories.

### Programming language clasification
OneVsRestClassifier was trained on StackOverflow posts with 10 tags to predict them.

### Finding the most relevant answer for the programming question
Starspace (facebook model) embeddings were trained on StackOverflow posts. All the posts were represented as vectors using these embeddings. For each tag a separate file with embeddings was created so that it wouldn't be necessary to load all the embeddings in the memory at once. User's question is also vectorized and most relevant answer is selected based on cosine similarity between the question and answers belonging to the predicted tag.

### Conversation
If the question is classified as non-programming, then conversational bot is activated.
I have used the idea of ChatBot from this repository: https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras
The chatbot is trained using teacher forcing. I used the pre-trained weights and fine-tuned the model on my own data. Also I rewrote most of the code, so it would be easier to use and understand. Most of the model's parameters are set in settings.ini file.

### Additional functionality
I decided to include some more functionality.

* Bot can give weather forecast for 5 days. I use for this openweathermap api;
* Bot can show the latest tweet of a certain user using twitter api;
* Bot can give a give a random fact about current date using http://numbersapi.com/;

### Bot limitations
Originally this bot was hosted on t2.micro tier of Amazon EC2, which implied quite limited resources. This is the reason that embeddings for each tag were saved separately to limit memory usage. Currently bot is hosted on t2.medium tier so that Keras model would have enough memory. The quality of model could be better, but it requires a lot of resources.

### Files
dialogue_manager.py - generates an answer for the user's input.

main_bot.py - main functionality of the bot - receiving user's input and sending the answer.

utils.py - additional functions for dealing with data.

settings.ini - model parameters.

settings_secret.ini - twitter and telegram tokens, paths to files.

data folder - contains embeddings and pickled models.

thread_embeddings_by_tags - embeddings for stackoverflow posts.

processer.py - processing data for training with chatbot.

chatbot.py - chatbot on keras.

Files in these two folders are too big to be uploaded on github. Also I'm not sure I may upload them as they are a part of coursera course. If you with to make something similar - join the course, please.

## Link to the bot
http://t.me/amlnlpbot

## Training the bot on your own data

It is possible to train the bot on your own data.
Data can be in one or several files with each utterance on a separate line. If utteraces are marked by names of some special symbols, add them to to_spaces list, so that they will be replaces by spaces. The model will perform better if dialogues have several lines, as in thic case more questions will have context.

If you want to train the model from scratch, you can do the following:

```python
from processer import DataProcesser
from chatbot import Chatbot

# list of files to use
list_of_files = []
processer = DataProcesser()
processer.process_text_initial(list_of_files)

bot = Chatbot()
bot.train
```

In this case a new vocabulary will be created based on your data. You'll need to download Glove embeddings (or some other embeddings) and define them in settings.ini file.
I'd recommend training for at least 100 epochs.
An important point is that the model requires a lot of memory due to it's architecture and teacher forcing. If you get memory error, set num_subsets in settings.ini to a higher value.

If you want to fine-tune the model on additional data, use `add_data` method of `DataProcesser()`. Also you'll need to download pre-trained weights - file my_model_weights20.h5 [here](https://www.dropbox.com/sh/o0rze9dulwmon8b/AAA6g6QoKM8hBEHGst6W4JGDa?dl=0)