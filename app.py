from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from datasets import load_dataset
import cassio
import os
import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader

ASTRA_DB_ID = "d02e6012-bd41-4f08-a5f7-72f23782509c"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:TgKenWfUaHuKhKLqsuHplBOm:061dbee39cabbea41e0b9fee23c92d5a8b21d3e1caee1e601eff15f343a7e8ef"

OPENAI_API_KEY = "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"

pdfreader = PdfReader('IPC cases.pdf')

from typing_extensions import Concatenate
#read text from file

raw_text = ''
for i, page in enumerate(pdfreader.pages):
  content = page.extract_text()
  if content:
    raw_text += content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key = OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa",
    session=None,
    keyspace=None,
)

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)



astra_vector_store.add_texts(texts)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

st.set_page_config(page_title="IPC query")
st.header("IntellectFlow: Case Conversations")

query_text = st.text_input("Type your query here: ")

answer = astra_vector_index.query(query_text, llm=llm).strip()

submit = st.button("Generate")

if submit:
  st.subheader("Here's what you've asked for..")
  st.write(answer)