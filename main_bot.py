#!/usr/bin/env python3

import requests
import time
import configparser
import json

from requests.compat import urljoin
from dialogue_manager import DialogueManager


class BotHandler(object):
    """
    BotHandler is a class which implements all back-end of the bot.
    It has tree main functions:
        'get_updates' — checks for new messages
        'send_message' – posts new message to user
        'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        # sometimes there are errors with received data. This prevents crashing.
        try:
            resp = requests.get(urljoin(self.api_url, "getUpdates"), params).json()
        except json.decoder.JSONDecodeError:
            resp = {}
        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        if 'png' in text:
            # telegram needs a bit different syntax for sending images.
            data = {"chat_id": chat_id}
            files = {'photo': (text, open(text, "rb"))}
            return requests.post(urljoin(self.api_url, "sendPhoto"), data=data, files=files)

        else:
            params = {"chat_id": chat_id, "text": text}
            return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am bot. To get more information about me send 'help!' message."
        return self.dialogue_manager.generate_answer(question)


def is_unicode(text):
    return len(text) == len(text.encode())


def main():
    config = configparser.ConfigParser()
    config.read('settings.ini')

    token = config['TELEGRAM']['TOKEN']

    if not token:
        print("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
        return

    manager = DialogueManager(config)
    bot = BotHandler(token, manager)

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        answer = bot.get_answer(update["message"]["text"])
                        if 'png' in answer:
                            # means the answer is weather forecast. Send text and image name separately.
                            bot.send_message(chat_id, answer.split(';')[0])
                            bot.send_message(chat_id, answer.split(';')[1])

                        else:
                            bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)

if __name__ == "__main__":
    main()
