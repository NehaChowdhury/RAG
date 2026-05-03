# 🗣️ Voice & PDF Chat Assistant (RAG)

A powerful, interactive Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using both **Text** and **Voice**. Built with Streamlit, LangChain, and Groq.

---

## 🌟 Features

- **📄 PDF Multi-Upload**: Upload one or multiple PDF files to chat with.
- **🎙️ Voice Interaction**: 
  - **Speech-to-Text**: Ask questions using your voice with real-time **Transcribing** status.
  - **Text-to-Speech**: The assistant responds with both text and high-quality voice (autoplay) using `gTTS`.
- **🧠 Intelligent RAG**: Uses LangChain and FAISS for efficient document retrieval.
- **💡 Smart Fallback**: If an answer isn't in the PDF, the AI provides a general answer while explicitly stating it wasn't found in the document.
- **🔄 UI Feedback**: Real-time "Transcribing..." and "Generating..." indicators for a smoother experience.
- **⚡ High Performance**: Powered by Groq's Llama 3.1-8B model for lightning-fast responses.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/)
- **Inference Engine**: [Groq Cloud](https://groq.com/)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Voice**: 
  - `SpeechRecognition` (Google API for STT)
  - `gTTS` (Google Text-to-Speech for cloud-compatible audio)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ 
- A [Groq API Key](https://console.groq.com/keys)

### 2. Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/NehaChowdhury/RAG.git
cd RAG

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Running the App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is optimized for **Streamlit Community Cloud**.

1. **Push to GitHub**: Ensure your latest changes are pushed to your repository.
2. **Deploy**: Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. **Secrets**: Add your `GROQ_API_KEY` in the Streamlit Cloud dashboard under **Settings > Secrets**:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

---

## 📖 How to Use

1. **Upload**: Use the sidebar to upload one or more PDFs.
2. **Process**: Click the "Process PDF" button to index the documents.
3. **Chat**: 
   - Type your question in the chat input.
   - **OR** Click the microphone icon, record your question, and wait for the assistant to reply!
4. **Listen**: The assistant will automatically read out the answer and show a "Generating..." status while preparing it.

---

## 🛡️ Privacy & Security
This project uses a `.gitignore` file to ensure that your `.env` (API Keys) and `.venv` (Virtual Environment) folders are never uploaded to GitHub. 

---

## 🤝 Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
