import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    PyPDFLoader("Pdfs/a1957-20.pdf"),
    PyPDFLoader("Pdfs/A2013-30.pdf"),
    PyPDFLoader("Pdfs/Coal Mines Regulation 2017.pdf"),
    PyPDFLoader("Pdfs/The Payment of Wages (Mines) Rules, 1956.pdf"),
    PyPDFLoader("Pdfs/theminesact1952.pdf"),
    PyPDFLoader("Pdfs/r&r_policy 2008.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

persist_directory = 'Pdfs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
