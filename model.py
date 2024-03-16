from typing import List

from langchain.chains import RetrievalQA
from langchain.document_loaders import YoutubeLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


class VectorDB:
    """Class to manage document loading and vector database creation."""

    def __init__(self,
                 url: str = None,
                 ):
        self.url = url

    def create_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=100
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
                                           search_kwargs={"k": 5},
                                           )
        return RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory,
        )


class Summarizer:
    def __init__(self,
                 url: str = None,
                 ):
        self.url = url

    def get_summary(self):
        vector_db = VectorDB(url=self.url)
        chunks: list[Document]
        chunks, _ = vector_db.create_vector_db()

        model = ChatOpenAI(model="gpt-3.5-turbo-16k",
                           temperature=0.3
                           )
        summarize_chain = load_summarize_chain(llm=model,
                                               chain_type="refine",
                                               verbose=True
                                               )
        return summarize_chain.run(chunks)


class MapReduceSummarizer:
    def __init__(self,
                 url: str = None,
                 ):
        self.url = url

    def get_summary(self):
        vector_db = VectorDB(url=self.url)
        llm = ChatOpenAI(temperature=0)
        docs: list[Document]
        docs, _ = vector_db.create_vector_db()
        # Map
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        from langchain import hub

        map_prompt = hub.pull("rlm/map-prompt")
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Reduce
        reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Note we can also get this from the prompt hub, as noted above
        reduce_prompt = hub.pull("rlm/map-prompt")

        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )
        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
        return map_reduce_chain.run(split_docs)