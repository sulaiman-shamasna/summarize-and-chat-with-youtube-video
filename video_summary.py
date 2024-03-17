from model import VectorDB
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain import hub
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter


class Summary:
    def __init__(self,
                 url: str = None,
                 ):
        self.url = url
        self.vector_db = VectorDB(url=self.url)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo",
                              temperature=0.0,
                              )
        self.chunks: list[Document]
        self.chunks, _ = self.vector_db.create_vector_db()

    def get_refine_summary(self):
        summarize_chain = load_summarize_chain(llm=self.llm,
                                               chain_type="refine",
                                               verbose=True
                                               )
        # print('TYPE:', type(summarize_chain.run(self.chunks))) # str
        return summarize_chain.run(self.chunks)

    def get_map_reduce_summary(self):
        map_template = """
                        The following is a set of documents {docs}
                        Based on this list of docs, please identify the main themes 
                        Helpful Answer:
                        """
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        map_prompt = hub.pull("rlm/map-prompt")
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        reduce_template = """
                            The following is set of summaries: {docs}
                            Take these and distill it into a final, consolidated summary of the main themes. 
                            Helpful Answer:
                            """
        reduce_prompt = hub.pull("rlm/map-prompt")
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="docs"
        )
        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=10000,
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
        split_docs = text_splitter.split_documents(self.chunks)
        return map_reduce_chain.run(split_docs)
