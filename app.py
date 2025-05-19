
chiave = st.secrets["superkey"]

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

st.header("Sono un ChatBot figo")

from PIL import Image
logo = Image.open("immagine.jpg")
st.image(logo)

with st.sidebar:
  st.title("Scrivi pure quello che desideri: ")
  file = st.file_uploader("Carica il tuo file: ", type="pdf")

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo += pagina.extract_text()

    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,  # Numero di caratteri per chunk
        chunk_overlap=150,
        length_function=len
    )

    # Suddivisione del testo
    pezzi = testo_spezzato.split_text(testo)

    # Visualizzazione dei pezzi
    # st.write(pezzi)

    embeddings = OpenAIEmbeddings(openai_api_key=chiave)
    vectorstore = FAISS.from_texts(pezzi, embeddings)

    domanda = st.text_input("Chiedi al chatbot:")

    if domanda:
        st.write("Sto cercado le informazioni che mi hai richiesto...")
        rilevanti = vectorstore.similarity_search(domanda)

        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key = chiave,
            temperature = 1.0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prendi la domanda, individua i frammenti rilevanti,
        # passali all'LLM, genera la risposta
        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents = rilevanti, question = domanda)
        st.write(risposta)
