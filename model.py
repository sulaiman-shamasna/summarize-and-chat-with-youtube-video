from typing import List
from langchain.chains import RetrievalQA
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter


class VectorDB:
    """Class to manage document loading and vector database creation."""

    def __init__(self,
                 url: str = None,
                 ):
        self.url = url

    def create_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )

        loader = YoutubeLoader.from_youtube_url(self.url)
        chunks = text_splitter.split_documents(loader.load())

        return chunks, Chroma.from_documents(chunks, OpenAIEmbeddings())


class ConversationalRetrievalChain:
    """Class to manage the QA chain setup."""

    def __init__(self, model_name="gpt-3.5-turbo",
                 temperature=0.3,
                 callbacks=None,
                 url: str = None
                 ):
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        self.model_name = model_name
        self.temperature = temperature
        self.url = url

    def create_chain(self):
        model = ChatOpenAI(model_name=self.model_name,
                           temperature=self.temperature,
                           streaming=True,
                           callbacks=self.callbacks
                           )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        vector_db = VectorDB(url=self.url)
        _, chroma_db = vector_db.create_vector_db()
        retriever = chroma_db.as_retriever(search_type="similarity",
                                           search_kwargs={"k": 2},
                                           )
        return RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory,
        )
