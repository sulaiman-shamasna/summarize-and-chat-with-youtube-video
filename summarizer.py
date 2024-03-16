from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
# from app import URL, secret_key
import os
import openai

from dotenv import load_dotenv, find_dotenv

openai.api_key = load_dotenv(find_dotenv())

# Save credentials for gradio app
secret_key = os.getenv('OPENAI_API_KEY')

# Load Transcript
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=r4wLXNydzeY', language=["en", "en-US"])
transcript = loader.load()

# Split Transcript
splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=100)
chunks = splitter.split_documents(transcript)

# chunks, _ = VectorDB(url=URL).create_vector_db()
# Set up LLM
openai_api_key = secret_key
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.3)

# Summarize
summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
summary = summarize_chain.run(chunks)

print(f'SUMMARY:\n{summary}')

# Write summary to file
with open("summaryY.txt", "w") as f:
    f.write(summary)
