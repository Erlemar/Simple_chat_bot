## Basic NLP chat-bot

This chatbot was created based on the final project of this course: https://www.coursera.org/learn/language-processing/home/welcome

The main functionality of the bot is to distinguish two types of questions: questions related to programming and others.

### Distinguishing intent

At first I had two datasets: StackOverflow posts (programming questions for 10 languages) and dialogue phrases from movie subtitles (non-programming questions). TfidfVectorizer with LogisticRegression were used to build model to classify user's question into these categories.

### Programming language clasification
OneVsRestClassifier was trained on StackOverflow posts with 10 tags to predict them.

### Finding the most relevant answer for the programming question
Starspace (facebook model) embeddings were trained on StackOverflow posts. All the posts were represented as vectors using these embeddings. For each tag a separate file with embeddings was created so that it wouldn't be necessary to load all the embeddings in the memory at once. User's question is also vectorized and most relevant answer is selected based on cosine similarity between the question and answers belonging to the predicted tag.

### Conversation
If the question is classified as non-programming, then conversational bot is activated. I used [ChatterBot](#https://github.com/gunthercox/ChatterBot) as suggested. It has enough training data to have more or less adequate conversations and is continuously trained on new conversations.

### Additional functionality
I decided to include some more functionality.

* Bot can give weather forecast for 5 days. I use for this openweathermap api;
* Bot can show the latest tweet of a certain user using twitter api;
* Bot can give a give a random fact about current date using http://numbersapi.com/;

### Bot limitations
This bot is hosted on t2.micro tier of Amazon EC2, which implies quite limited resources. This is the reason that embeddings for each tag are saved separately to limit memory usage. Also I didn't use additional conversation corpus to train conversational bot or to make a seq2seq model by myself. These are the tasks for the future.

### Files
dialogue_manager.py - generates an answer for the user's input.

main_bot.py - main functionality of the bot - receiving user's input and sending the answer.

utils.py - additional functions for dealing with data.

settings.ini - twitter and telegram tokens, paths to files.

data folder - contains embeddings and pickled models.

thread_embeddings_by_tags - embeddings for stackoverflow posts.

Files in these two folders are too big to be uploaded on github. Also I'm not sure I may upload them as they are a part of coursera course. If you with to make something similar - join the course, please.

## Link to the bot
http://t.me/amlnlpbot