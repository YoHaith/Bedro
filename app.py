# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb requests fuzzywuzzy

import streamlit as st
import json
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup
import requests
from langchain.docstore.document import Document
from fuzzywuzzy import fuzz, process

load_dotenv()

def load_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def get_vectorstore_from_urls(urls):
    documents = []
    for url in urls:
        content = load_content(url)
        document = Document(page_content=content, metadata={'source': url})
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents([document])
        documents.extend(document_chunks)

    # ✅ NO arguments passed to OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_header():
    logo_url = "https://i.imgur.com/qVy8JRy.png"
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{logo_url}" alt="Logo" style="width:20%;">
            <h2 style="color: black;">Born in the Wilderness</h2>
            <h5 style="color: Grey;">Du hast Fragen? Chat mit uns (BETA v1.0)</h5>
        </div>
    """, unsafe_allow_html=True)

# Load links and product data
links = load_json("links.json")
products = load_json("products.json")

# app config
st.set_page_config(page_title="Chat with Online Store", page_icon="🛒")
display_header()

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls(links)

def find_best_matches(query, products, threshold=70):
    product_names = [product["name"] for product in products]
    best_matches = process.extract(query, product_names, scorer=fuzz.partial_ratio)
    filtered_matches = [match for match in best_matches if match[1] >= threshold]
    matched_products = []
    for match in best_matches:
        matched_name = match[0]
        for product in products:
            if product["name"] == matched_name:
                matched_products.append(product)
    return matched_products

def get_variant(query, products):
    best_matches = find_best_matches(query, products)
    if not best_matches:
        return None
    for product in best_matches:
        if fuzz.partial_ratio(query, product["link"]) > 70:
            return product
    return best_matches[0]

def get_unique_products(products):
    seen = set()
    unique_products = []
    for product in products:
        if product["name"] not in seen:
            unique_products.append(product)
            seen.add(product["name"])
    return unique_products

# user input
user_query = st.chat_input("Type your message here...")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
unique_products = get_unique_products(products)
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
            best_matches = find_best_matches(message.content, unique_products)
            for match in best_matches:
                variant = get_variant(message.content, [product for product in products if product["name"] == match["name"]])
                if variant:
                    st.image(variant["image"], width=200)
                    st.markdown(f"[Jetzt Kaufen]({variant['link']})", unsafe_allow_html=True)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
