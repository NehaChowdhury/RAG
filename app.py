import streamlit as st
import os
import tempfile
import base64
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from audio_recorder_streamlit import audio_recorder

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------- LOAD ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Voice Chat", layout="wide")
st.title("🗣️ Chat with PDF")

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


# ================= PDF PROCESSING =================
def extract_text(pdf_files):
    text = ""

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    text += page_text + "\n"

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    return text.strip()


def split_text(text):
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vectorstore(chunks):
    if not chunks:
        st.error("No text chunks found. PDF may be scanned or empty.")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embedding=embeddings)


# ================= VOICE → TEXT =================
def voice_to_text(audio_bytes):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    try:
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)

    except:
        return None
    finally:
        os.remove(path)


# ================= TEXT → VOICE =================
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name

    tts.save(path)

    with open(path, "rb") as f:
        audio = f.read()

    os.remove(path)
    return audio


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f"""
    <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)


# ================= RAG =================
def ask_pdf(query):

    if st.session_state.vectorstore is None:
        return "Please upload and process a PDF first."

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    retriever = st.session_state.vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question using the provided context from the PDF. "
         "If the information is NOT in the context, you MUST start your response with "
         "'This information is not in the PDF, but...' and then provide the answer using your own general knowledge. "
         "If the information IS in the context, just provide the answer directly.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    chain = (
        {"context": retriever | format_docs,
         "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)


# ================= SIDEBAR =================
with st.sidebar:
    st.header("Upload PDF")

    pdf_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process PDF"):
        if pdf_files:

            with st.spinner("Processing PDF..."):

                text = extract_text(pdf_files)

                if not text:
                    st.error("No text extracted. PDF might be scanned.")
                else:
                    chunks = split_text(text)

                    if not chunks:
                        st.error("Text splitting failed.")
                    else:
                        vectorstore = create_vectorstore(chunks)

                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.success(f"PDF processed! {len(chunks)} chunks created")

        else:
            st.warning("Please upload at least one PDF")


# ================= CHAT HISTORY =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("audio"):
            if msg == st.session_state.chat_history[-1]:
                autoplay_audio(msg["audio"])


# ================= INPUT UI =================
col1, col2 = st.columns([5, 1])

with col1:
    user_text = st.chat_input("Ask something about your PDF...")

with col2:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x"
    )


user_query = None

if user_text:
    user_query = user_text

if audio_bytes:
    audio_hash = hash(audio_bytes)

    if st.session_state.last_audio_hash != audio_hash:
        st.session_state.last_audio_hash = audio_hash

        with st.spinner("Transcribing voice..."):
            text = voice_to_text(audio_bytes)

        if text:
            user_query = text


# ================= USER MESSAGE =================
if user_query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    st.rerun()


# ================= AI RESPONSE =================
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
):

    query = st.session_state.chat_history[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = ask_pdf(query)
            audio = text_to_speech(response)
        st.write(response)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "audio": audio
    })

    st.rerun()