
import os
import openai
import sys
sys.path.append('../..')

from langchain_openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

persist_directory = 'Pdfs/chroma/'
embedding = OpenAIEmbeddings()

# Load the persisted Chroma vector database with the provided embedding function
vectordb = Chroma(
     persist_directory=persist_directory,
     embedding_function=embedding  # Add this line
)

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "What must the competent authority do when taking action under sub-section (3) of section 4, and how are disputes regarding compensation handled?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


result = qa_chain.invoke({"query": question})
print(result['result'])