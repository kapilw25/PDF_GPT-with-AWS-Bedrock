import json
import os
import sys
import boto3
import streamlit as st
import warnings

# use Titan embeddings model to genrate embedding from PDFs
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector embedding and Vector Store
from langchain.vectorstores import FAISS

# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

# Bedrock clients
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    # characteristics splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

# Vector (Titan) embedding and Vector Store (FAISS)
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings,
    )
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    # create the Anthropic model
    llm = Bedrock(
        model_id = "ai21.j2-mid-v1",
        client = bedrock,
        model_kwargs= {'maxTokens': 512} 
    )
    return llm

def get_llama2_llm():
    llm = Bedrock(
        model_id = "meta.llama2-70b-chat-v1",
        client = bedrock,
        model_kwargs= {'max_gen_len': 512} 
    )
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but usse atleast summarize with 250 words with detailed explaantions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

def truncate_prompt(prompt, max_length=2048):
    return prompt[:max_length]

def get_response_llm(llm, vectorstore_faiss, query):
    truncated_query = truncate_prompt(query)  # Truncate the query to ensure it's within the token limit
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": truncated_query})
    return answer['result']


# create Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock emoji:books:")
    
    user_question = st.text_input("Ask a question from the PDF files")
    
    
    with st.sidebar:
        st.title("Update or create vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Updating Vector Store"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector Store updated successfully")
                
    if st.button("Claude Output"):
        with st.spinner("Getting Claude Output"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done: Claude Output")
            
    if st.button("Llama2 Output"):
        with st.spinner("Getting Llama2 Output"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                                           
            llm = get_llama2_llm()
            
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done: Llama2 Output")
            
if __name__ == "__main__":
    main()