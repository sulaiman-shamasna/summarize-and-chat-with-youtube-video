import os
from queue import Queue, Empty
from threading import Thread
from typing import Any

import gradio as gr
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks.base import BaseCallbackHandler

from footer import DESCRIPTION
from model import *

from summarizer import summary

URL: str = 'https://www.youtube.com/watch?v=r4wLXNydzeY'
# URL: str = 'https://www.youtube.com/watch?v=HJXWpqpcHik'
# Load OpenAI API key from environment variables
openai.api_key = load_dotenv(find_dotenv())

# Save credentials for gradio app
secret_key = os.getenv('OPENAI_API_KEY')
username = 'username'

# Initialize a queue for communication between threads
q: Queue = Queue()
job_done = object()


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM (LangChain) responses to a queue."""

    def __init__(self, q: Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Callback method to add new token to the queue."""
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> bool:
        """Callback method called when LLM interaction ends."""
        return self.q.empty()


# Define a list of callbacks to be used by the LLM (LangChain)
callbacks = [QueueCallback(q)]


def answer(question: str) -> None:
    """Function to predict response to a given question."""

    def task():
        # response = conversation.predict(input=question)
        qa_chain = ConversationalRetrievalChain(callbacks=callbacks,
                                                url=URL).create_chain()
        response = qa_chain({"query": question})
        q.put(job_done)

    t = Thread(target=task)
    t.start()


def validate_youtube_url(url):
    # Implement URL validation logic here
    # Return True if the URL is valid, False otherwise
    # For example, you can use regular expressions to validate the URL
    # For simplicity, let's assume all URLs are valid for now
    return True


# Create a gradio block
blocker = gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.green))

# Define the Gradio interface
with blocker as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab("Summary"):
        gr.Textbox(summary)

    with gr.Tab("QA"):

        chatbot = gr.Chatbot()
        message = gr.Textbox()
        clear = gr.Button("Clear")


        def user(user_message: str, history: list) -> tuple:
            """Function to process user input."""
            return "", history + [[user_message, None]]


        def bot(history: list) -> Any:
            """Function to handle bot responses."""
            question = history[-1][0]
            print("Question: ", question)
            history[-1][1] = ""
            answer(question=question)
            while True:
                try:
                    next_token = q.get(True)
                    if next_token is job_done:
                        break
                    history[-1][1] += next_token
                    yield history
                except Empty:
                    continue


        # Define Gradio components and interactions
        message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
demo.queue()
demo.launch()
# demo.launch(auth=(username, secret_key),
#             auth_message='Enter your username and passward, please!',
#             server_name='localhost',
#             server_port=7860,
#             )
