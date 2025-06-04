import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # FIXED: Importazione corretta
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import base64
import csv
import pandas as pd
import plotly.express as px
import os

# -------------------------
# CONFIGURAZIONE PAGINA
# -------------------------
st.set_page_config(
    page_title="Chatbot GTA San Andreas",
    page_icon=":video_game:",
    layout="wide"
)

# -------------------------
# FUNZIONI HELPER
# -------------------------
def get_base64_of_image(image_path):
    """Converte immagine in base64 con gestione errori"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        else:
            st.warning(f"Immagine {image_path} non trovata")
            return None
    except Exception as e:
        st.error(f"Errore nel caricamento dell'immagine {image_path}: {str(e)}")
        return None

def load_image_safe(image_path, default_width=150):
    """Carica immagine con gestione errori"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"Immagine {image_path} non trovata")
            return None
    except Exception as e:
        st.error(f"Errore nel caricamento di {image_path}: {str(e)}")
        return None

# -------------------------
# CONFIGURAZIONE BACKGROUNDS E PERSONAGGI
# -------------------------
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

personaggi = {
    "CJ": {
        "immagine": "cj.png",
        "nome": "Carl 'CJ' Johnson",
        "colore": "#00aa00"
    },
    "Big Smoke": {
        "immagine": "bigsmoke.png",
        "nome": "Melvin 'Big Smoke' Harris",
        "colore": "#808080"
    },
    "Tenpenny": {
        "immagine": "tenpenny.webp",
        "nome": "Frank Tenpenny",
        "colore": "#0077ff"
    },
    "Sweet": {
        "immagine": "Sweet.webp",
        "nome": "Sean 'Sweet' Johnson",
        "colore": "#556b2f"
    }
}

# -------------------------
# CSS STYLES
# -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');
    
    /* Sidebar stile GTA */
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
    
    .unifraktur {
        font-family: 'UnifrakturCook', cursive;
        font-size: 48px;
        color: white;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    # Easter egg
    user_input = ""
    if user_input.strip().lower() == "big smoke order":
        if load_image_safe("bigsmoke.png"):
            st.image("bigsmoke.png", width=150)
        st.markdown("## 🍔 Ordine di Big Smoke")
        media_type = st.radio("🎬 Mostra la scena come:", ["GIF", "Video"])
        if media_type == "GIF":
            if os.path.exists("gtagif.gif"):
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

    # Mostra personaggio selezionato
    personaggio_img = load_image_safe(personaggi[bot_style]["immagine"])
    if personaggio_img:
        st.image(personaggi[bot_style]["immagine"], width=150)
    
    st.markdown(
        f"<h3 style='color: {personaggi[bot_style]['colore']};'>{personaggi[bot_style]['nome']}</h3>",
        unsafe_allow_html=True
    )

    # Selezione location
    location = st.selectbox("🌍 Ambientazione", list(backgrounds.keys()))

    # Musica
    st.markdown("🎵 Ascolta le radio del gioco:")
    if os.path.exists(backgrounds[location]["audio"]):
        st.audio(backgrounds[location]["audio"])
    else:
        st.warning(f"File audio {backgrounds[location]['audio']} non trovato")
    st.markdown(f"**In riproduzione:** {backgrounds[location]['radio_name']}")

    st.markdown("---")
    st.caption("GTA ChatBot v1.0 — Made in Grove Street 🟢")

# -------------------------
# SFONDO DINAMICO
# -------------------------
bg_base64 = get_base64_of_image(backgrounds[location]["image"])
if bg_base64:
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

# -------------------------
# LOGO E HEADER
# -------------------------
logo = load_image_safe("gtalogo.png")
if logo:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, use_container_width=True)

st.markdown('<div class="gta-header">Fai una domanda sul videogioco:</div>', unsafe_allow_html=True)

# -------------------------
# ELABORAZIONE FILE PDF
# -------------------------
# FIXED: Ripristino file uploader invece di file hardcoded
file = st.file_uploader("Carica il manuale PDF", type=["pdf"])

# Fallback per file locale se presente
if file is None and os.path.exists("gtamanuale.pdf"):
    try:
        with open("gtamanuale.pdf", "rb") as f:
            file = f.read()
    except Exception as e:
        st.error(f"Errore nel caricamento del file locale: {str(e)}")

if file is not None:
    try:
        # Inizializzazione session state
        if "domanda_inviata" not in st.session_state:
            st.session_state.domanda_inviata = ""
        if "domanda" not in st.session_state:
            st.session_state.domanda = ""

        # Lettura PDF
        if isinstance(file, bytes):
            # File locale
            import io
            pdf_file = io.BytesIO(file)
            testo_letto = PdfReader(pdf_file)
        else:
            # File uploadato
            testo_letto = PdfReader(file)

        testo = ""
        for pagina in testo_letto.pages:
            testo += pagina.extract_text()

        if not testo.strip():
            st.error("Il PDF sembra essere vuoto o non leggibile.")
        else:
            # Splitting del testo
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "!", "?"],
                length_function=len
            )
            chunks = splitter.split_text(testo)

            # Verifica chiave API
            if "superkey" not in st.secrets:
                st.error("⚠️ Chiave API OpenAI non configurata nei secrets!")
                st.stop()

            superkey = st.secrets["superkey"]

            try:
                # FIXED: Uso della nuova OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(api_key=superkey)
                vector_store = FAISS.from_texts(chunks, embeddings)

                def invia():
                    st.session_state.domanda_inviata = st.session_state.domanda
                    st.session_state.domanda = ""

                st.text_input("❓ Fai una domanda sul videogioco:", key="domanda", on_change=invia)
                domanda = st.session_state.get("domanda_inviata", "")

                if domanda:
                    with st.spinner("🔍 Sto cercando nel manuale..."):
                        try:
                            risultati = vector_store.similarity_search(domanda, k=10)

                            llm = ChatOpenAI(
                                api_key=superkey,  # FIXED: parametro corretto
                                temperature=0.3,
                                max_tokens=800,
                                model_name="gpt-3.5-turbo"
                            )

                            chain = load_qa_chain(llm, chain_type="stuff")
                            risposta = chain.run(input_documents=risultati, question=domanda)

                            st.success("✅ Risposta trovata:")
                            st.write(risposta)

                        except Exception as e:
                            st.error(f"Errore nella ricerca: {str(e)}")

            except Exception as e:
                st.error(f"Errore nell'inizializzazione delle embeddings: {str(e)}")

    except Exception as e:
        st.error(f"Errore nell'elaborazione del PDF: {str(e)}")

st.markdown("---")

# -------------------------
# DATI DI VENDITA
# -------------------------
try:
    if os.path.exists("videogames.csv"):
        dati = pd.read_csv("videogames.csv", encoding="latin1", sep=";")
        
        st.markdown('<div class="unifraktur">📊 Dashboard vendite:</div>', unsafe_allow_html=True)

        if "vendite_milioni" in dati.columns:
            vendite_milioni = int(dati["vendite_milioni"].sum()) * 1000000
            media_transazione = round(dati["vendite_milioni"].mean(), 2)

            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("Vendite totali:")
                st.subheader(f"US $ {vendite_milioni:,}")

            with right_col:
                st.subheader("Ricavi Aggiornati al 2025:")
                if "ricavi" in dati.columns:
                    totale_ricavi = dati["ricavi"].sum() * 1000000
                    st.subheader(f"US $ {totale_ricavi:,}")
                else:
                    st.subheader("Dati non disponibili")

            st.markdown("---")

            # Grafico vendite per genere
            if "genere" in dati.columns:
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

            # Heatmap dati di vendita
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
        else:
            st.warning("Colonna 'vendite_milioni' non trovata nel CSV")
    else:
        st.warning("File videogames.csv non trovato")

except Exception as e:
    st.error(f"Errore nel caricamento dei dati di vendita: {str(e)}")
