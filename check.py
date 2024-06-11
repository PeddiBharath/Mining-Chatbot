
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


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "What action must the Central Government take after considering the report under section 8 regarding the acquisition of land or rights over such land?"
result = qa.invoke({"question": question})

print(result)

question = "Is there a time limit for making a declaration regarding the acquisition of land or rights over such land covered by a notification under sub-section (1) of section 7 issued after the commencement of the Coal Bearing Areas (Acquisition and Development) Amendment and Validation Act, 1971?"
result = qa.invoke({"question": question})

print(result)

