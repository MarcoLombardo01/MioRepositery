import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import base64
import csv
import pandas as pd
import plotly.express as px
import traceback

try:

    st.set_page_config(
        page_title="Chatbot GTA San Andreas",
        page_icon=":video_game:",
        layout="wide"
    )

    backgrounds = {
        "Los Santos": {
            "image": "lossantos.jpg",
            "radio_name": "Radio Los Santos",
            "audio": "radiolossantos.mp3"
        },
        "Countryside": {
            "image": "countryside.jpg",
            "radio_name": "K-Rose",
            "audio": "krose.mp3"
        },
        "San Fierro": {
            "image": "sanfierro.jpg",
            "radio_name": "Radio X",
            "audio": "radiox.mp3"
        },
        "Las Venturas": {
            "image": "lasventuras.jpg",
            "radio_name": "K-Jah West",
            "audio": "kjahwest.mp3"
        }
    }

    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #111;
        color: white;
        padding: 20px;
        font-family: 'Trebuchet MS', sans-serif;
        border-right: 4px solid #2ecc71;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2ecc71;
        font-weight: bold;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] button {
        background-color: #2ecc71 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .stRadio label {
        color: white;
        font-weight: bold;
        font-size: 18px;
    }
    .gta-name {
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        font-family: 'Trebuchet MS', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        personaggi = {
            "CJ": {"immagine": "cj.png", "nome": "Carl 'CJ' Johnson", "colore": "#00aa00"},
            "Big Smoke": {"immagine": "bigsmoke.png", "nome": "Melvin 'Big Smoke' Harris", "colore": "#808080"},
            "Tenpenny": {"immagine": "tenpenny.webp", "nome": "Frank Tenpenny", "colore": "#0077ff"},
            "Sweet": {"immagine": "Sweet.webp", "nome": "Sean 'Sweet' Johnson", "colore": "#556b2f"}
        }

        user_input = ""
        if user_input.strip().lower() == "big smoke order":
            st.image("bigsmoke.png", width=150)
            st.markdown("## 🍔 Ordine di Big Smoke")
            media_type = st.radio("🎬 Mostra la scena come:", ["GIF", "Video"])
            if media_type == "GIF":
                st.image("gtagif.gif")
            else:
                st.video("https://youtu.be/A6g0mPo-uJM")
            st.markdown("""
                > *I'll have:*
                - 2️⃣ Number 9
                - 1️⃣ Number 9 Large
                - 1️⃣ Number 6 with extra dip
                - 1️⃣ Number 7
                - 2️⃣ Number 45 (one with cheese)
                - 1️⃣ Large Soda 🥤
            """)
            st.success("Ordine speciale attivato!")
            bot_style = "Big Smoke"
        else:
            bot_style = st.radio("Scopri i protagonisti:", list(personaggi.keys()))

        st.image(personaggi[bot_style]["immagine"], width=150)
        st.markdown(
            f"<h3 style='color: {personaggi[bot_style]['colore']};'>{personaggi[bot_style]['nome']}</h3>",
            unsafe_allow_html=True
        )

        location = st.selectbox("🌍 Ambientazione", list(backgrounds.keys()))
        st.markdown("🎵 Ascolta le radio del gioco:")
        st.audio(backgrounds[location]["audio"])
        st.markdown(f"**In riproduzione:** {backgrounds[location]['radio_name']}")
        st.markdown("---")
        st.caption("GTA ChatBot v1.0 — Made in Grove Street 🟢")

    def get_base64_of_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    bg_base64 = get_base64_of_image(backgrounds[location]["image"])

    st.markdown(
        f"""
        <style>
        [data-testid="stApp"] {{
            background-image: url("data:image/jpg;base64,{bg_base64}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    logo = Image.open("gtalogo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, use_container_width=True)

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');
        .main .block-container,
        .main .block-container h1,
        .main .block-container h2,
        .main .block-container h3,
        .main .block-container p,
        .main .block-container div,
        .stMarkdown,
        .stSuccess,
        .stInfo,
        .stSubheader {
            font-family: 'UnifrakturCook', cursive !important;
            color: white !important;
            text-shadow: 2px 2px 4px #000000 !important;
        }
        .gta-header {
            font-family: 'UnifrakturCook', cursive;
            font-size: 36px;
            color: white;
            text-shadow: 2px 2px 4px #000000;
            margin-bottom: 20px;
        }
        </style>
        <div class="gta-header">Fai una domanda sul videogioco:</div>
    """, unsafe_allow_html=True)

    file = "gtamanuale.pdf"
    if file is not None:
        testo_letto = PdfReader(file)
        testo = ""
        for pagina in testo_letto.pages:
            testo = testo + pagina.extract_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?"],
            length_function=len
        )
        chunks = splitter.split_text(testo)

        superkey = st.secrets["superkey"]
        embeddings = OpenAIEmbeddings(openai_api_key=superkey)
        vector_store = FAISS.from_texts(chunks, embeddings)

        def invia():
            st.session_state.domanda_inviata = st.session_state.domanda
            st.session_state.domanda = ""

        st.text_input("❓ Fai una domanda sul videogioco:", key="domanda", on_change=invia)
        domanda = st.session_state.get("domanda_inviata", "")

        if domanda:
            st.info("🔍 Sto cercando nel manuale...")
            risultati = vector_store.similarity_search(domanda, k=10)

            llm = ChatOpenAI(
                openai_api_key=superkey,
                temperature=0.3,
                max_tokens=800,
                model_name="gpt-3.5-turbo-0125"
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            risposta = chain.run(input_documents=risultati, question=domanda)

            st.success("✅ Risposta trovata:")
            st.write(risposta)

    st.markdown("---")

    dati = pd.read_csv("videogames.csv", encoding="latin1", sep=";")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:700&display=swap');
        .unifraktur {
            font-family: 'UnifrakturCook', cursive;
            font-size: 48px;
            color: white;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="unifraktur">📊 Dashboard vendite:</div>', unsafe_allow_html=True)

    vendite_milioni = int(dati["vendite_milioni"].sum()) * 1000000
    media_transazione = round(dati["vendite_milioni"].mean(), 2)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Vendite totali:")
        st.subheader(f"US $ {vendite_milioni:,}")

    with right_col:
        st.subheader("Ricavi Aggiornati al 2025:")
        totale_ricavi = dati["ricavi"].sum() * 1000000
        st.subheader(f"US $ {totale_ricavi:,}")

    st.markdown("---")

    dati_cat = dati.groupby(by=["genere"], as_index=False)["vendite_milioni"].sum().sort_values(by="vendite_milioni")
    st.markdown('<div class="unifraktur" style="font-size: 36px; margin-top: 30px;">🎮 Vendite per genere</div>', unsafe_allow_html=True)

    fig = px.bar(dati_cat,
                 y="genere",
                 x="vendite_milioni",
                 text=[f"$ {x:.1f}" for x in dati_cat["vendite_milioni"]],
                 orientation="h")

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray", title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(gridcolor="lightgray", title_font=dict(color="black"), tickfont=dict(color="black")),
        font=dict(color="black")
    )

    fig.update_traces(
        marker_color='lightblue',
        textfont=dict(color="black")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    with st.expander("Heatmap dati di vendita"):
        styled_data = dati.style.background_gradient(cmap="Blues").set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black')]},
            {'selector': 'td', 'props': [('background-color', 'white'), ('color', 'black')]},
            {'selector': 'table', 'props': [('background-color', 'white')]}
        ])

        st.write(styled_data)

        csv = dati.to_csv(index=False).encode("utf-8")
        st.download_button("Scarica i dati",
                          data=csv,
                          file_name="Dati di Vendita GTA San Andreas.csv",
                          mime="text/csv",
                          help="Clicca qui per scaricare")

except Exception as e:
    st.error("❌ Errore nell'app:")
    st.text(traceback.format_exc())
