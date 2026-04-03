import streamlit as st # type: ignore
import os
import tempfile
import base64
import speech_recognition as sr # type: ignore
import pyttsx3 # type: ignore
from dotenv import load_dotenv # type: ignore
from PyPDF2 import PdfReader # type: ignore
from audio_recorder_streamlit import audio_recorder # type: ignore
from io import BytesIO

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore

# Load environment variables
load_dotenv()

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit config
st.set_page_config(page_title="Voice & PDF Chat App", layout="wide")
st.title("🗣️ Chat with your PDF (Text + Voice)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None


# ---------- PDF FUNCTIONS ----------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# ---------- AUDIO FUNCTIONS ----------

def transcribe_audio(audio_bytes):

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text

    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."

    except sr.RequestError:
        return "Speech recognition service unavailable."

    finally:
        os.remove(temp_path)


def generate_audio(text):
    """
    Generate audio using pyttsx3 (offline TTS).
    Since pyttsx3 saves to file, we use a temporary file and read it back.
    """
    engine = pyttsx3.init()
    
    # Configure voice properties if needed
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id) # Set a specific voice
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_path = temp_audio.name
    
    try:
        # pyttsx3 save_to_file is synchronous by default or needs runAndWait
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)


# ---------- RAG FUNCTION ----------

def process_query(user_query):

    if st.session_state.vectorstore is None:
        return "Please upload and process a PDF first."

    try:
        llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)
    except Exception as e:
        return f"LLM initialization error: {e}"

    retriever = st.session_state.vectorstore.as_retriever()

    system_prompt = (
        "You are a helpful assistant. "
        "Strictly follow these rules:\n"
        "1. If the retrieved context contains the answer, provide it directly without any disclaimer.\n"
        "2. If the retrieved context DOES NOT contain the answer, start your response with exactly: "
        "'This information is not in the PDF, but...' and then provide the answer using your own knowledge.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        if not docs:
            return "No context available."
        return "\n\n".join(doc.page_content for doc in docs)

    # Use a dummy retriever if vectorstore is missing
    context_fetcher = (retriever | format_docs) if st.session_state.vectorstore else (lambda x: "No context available.")

    rag_chain = (
        {"context": context_fetcher, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_query) # type: ignore

    return response


# ---------- SIDEBAR ----------

with st.sidebar:

    st.subheader("Upload PDF")

    pdf_docs = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process"):

        if pdf_docs:

            with st.spinner("Processing PDFs..."):

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.vectorstore = vectorstore

                st.success("PDF processed successfully!")

        else:
            st.warning("Please upload at least one PDF")


# ---------- CHAT DISPLAY ----------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message.get("audio"):
            # If it's the most recent AI response, autoplay it
            if message == st.session_state.chat_history[-1] and message["role"] == "assistant":
                autoplay_audio(message["audio"])


# ---------- INPUT AREA ----------

col1, col2 = st.columns([5, 1])

with col1:
    user_text = st.chat_input("Ask a question about your document...")

with col2:
    st.write("Speak")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x"
    )


user_query = None

if user_text:
    user_query = user_text


# Voice input
if audio_bytes:

    audio_hash = hash(audio_bytes)

    if st.session_state.last_processed_audio != audio_hash:

        st.session_state.last_processed_audio = audio_hash

        with st.spinner("Transcribing audio..."):
            user_query = transcribe_audio(audio_bytes)


# Add user query to chat
if user_query:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    st.rerun()


# Generate AI response
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
):

    last_query = st.session_state.chat_history[-1]["content"]

    response = process_query(last_query)

    with st.spinner("Generating voice..."):
        audio_response = generate_audio(response)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "audio": audio_response
    })

    st.rerun()