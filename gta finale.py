
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


# -------------------------
# CONFIGURAZIONE PAGINA
# -------------------------
st.set_page_config(
    page_title="Chatbot GTA San Andreas",
    page_icon=":video_game:",
    layout="wide"
)

#Immagini sfondo per ambientazioni

# Immagini sfondo e audio radio per ambientazioni
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



# -------------------------
# SIDEBAR - UPLOAD FILE
# -------------------------
# SIDEBAR STILE GTA COLORI
st.markdown(
    """
    <style>
    /* Sidebar stile GTA */
    [data-testid="stSidebar"] {
        background-color: #111; /* Nero tipico di GTA */
        color: white;
        padding: 20px;
        font-family: 'Trebuchet MS', sans-serif;
        border-right: 4px solid #2ecc71; /* Bordo verde neon */
    }

    /* Titoli sidebar */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2ecc71;
        font-weight: bold;
        text-transform: uppercase;
    }

    /* Pulsanti della sidebar */
    [data-testid="stSidebar"] button {
        background-color: #2ecc71 !important; /* Verde neon per pulsanti */
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid white;
    }

    </style>
    """,
    unsafe_allow_html=True
)
#colore bianco scopri personaggio

st.markdown(
    """
    <style>
    /* Testo "Scopri i protagonisti" in bianco */
    .stRadio label {
        color: white;
        font-weight: bold;
        font-size: 18px;
    }

    /* Nomi personaggi in bianco */
    .gta-name {
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        font-family: 'Trebuchet MS', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Nomi personaggi
with st.sidebar:
    personaggi = {
    "CJ": {
        "immagine": "cj.png",
        "nome": "Carl 'CJ' Johnson",
        "colore": "#00aa00"  # verde
    },
    "Big Smoke": {
        "immagine": "bigsmoke.png",
        "nome": "Melvin 'Big Smoke' Harris",
        "colore": "#808080"  # grigio
    },
    "Tenpenny": {
        "immagine": "tenpenny.webp",
        "nome": "Frank Tenpenny",
        "colore": "#0077ff"  # blu elettrico
    },
    "Sweet": {
        "immagine": "Sweet.webp",
        "nome": "Sean 'Sweet' Johnson",
        "colore": "#556b2f"  # verde militare
    }
}

# Selezione del personaggio
    bot_style = st.selectbox("Scegli un personaggio:", list(personaggi.keys()))

    # Mostra personaggio selezionato
    st.image(personaggi[bot_style]["immagine"], width=150)
    st.markdown(
    f"<h3 style='color: {personaggi[bot_style]['colore']};'>{personaggi[bot_style]['nome']}</h3>",
    unsafe_allow_html=True
)

    # Selezione location
    location = st.selectbox("🌍 Ambientazione", list(backgrounds.keys()))

    # Musica
    st.markdown("🎵 Ascolta le radio del gioco:")
    st.audio(backgrounds[location]["audio"])
    st.markdown(f"**In riproduzione:** {backgrounds[location]['radio_name']}")


    st.markdown("---")
    st.caption("GTA ChatBot v1.0 — Made in Grove Street 🟢")

# -------------------------
# CSS DINAMICO PER SFONDO
# -------------------------
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
    ...
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------
# LOGO E HEADER
# -------------------------


logo = Image.open("gtalogo.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, use_container_width=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');

    /* Font GTA per tutta la pagina principale */
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

# -------------------------
# ELABORAZIONE FILE
# -------------------------
#file = st.file_uploader("gtamanuale.pdf", type=["pdf"])
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

    # Inserisci qui la tua chiave API OpenAI
    superkey = "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"

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

# -------------------------
# DATI DI VENDITA
# -------------------------
dati = pd.read_csv("videogames.csv", encoding="latin1", sep=";")
print(dati.columns.tolist())

# Inserimento del CSS personalizzato per il font
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

# Titolo personalizzato
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

# -------------------------
# GRAFICO VENDITE PER GENERE
# -------------------------
dati_cat = dati.groupby(by=["genere"], as_index=False)["vendite_milioni"].sum().sort_values(by="vendite_milioni")

st.markdown('<div class="unifraktur" style="font-size: 36px; margin-top: 30px;">🎮 Vendite per genere</div>', unsafe_allow_html=True)
fig = px.bar(dati_cat,
             y="genere",
             x="vendite_milioni",
             text=[f"$ {x:.1f}" for x in dati_cat["vendite_milioni"]],
             orientation="h")

# Modifiche per sfondo bianco e testo nero
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgray", title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(gridcolor="lightgray", title_font=dict(color="black"), tickfont=dict(color="black")),
    font=dict(color="black")
)

# Colore delle barre e testo
fig.update_traces(
    marker_color='lightblue',
    textfont=dict(color="black")
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------
# HEATMAP DATI DI VENDITA
# -------------------------
with st.expander("Heatmap dati di vendita"):
    # Applicazione dello stile con sfondo bianco e testo nero
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
