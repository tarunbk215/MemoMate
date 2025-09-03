import os
import streamlit as st
import fitz  # PyMuPDF
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
from dotenv import load_dotenv

# ==================== LOAD ENV ====================
load_dotenv()  # ✅ loads .env file automatically

# ==================== PAGE CONFIG (must be first Streamlit command) ====================
st.set_page_config(
    page_title="StudyMate - PDF Summarizer & Q/A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HUGGING FACE LOGIN ====================
hf_token = os.getenv("HF_API_KEY")
if hf_token:
    login(hf_token)  # ✅ correct usage
else:
    st.warning("⚠️ Hugging Face API key not found! Please set HF_API_KEY in your .env file.")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=-1  # CPU
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, embedder

summarizer, embedder = load_models()

# ==================== PDF EXTRACTION ====================
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text.strip()

# ==================== TEXT SPLITTING ====================
def split_text_into_chunks(text, max_words=300):  # ✅ smaller chunks
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# ==================== SUMMARIZATION ====================
def summarize_large_text(text):
    if len(text.split()) < 50:
        return "Document too short to summarize."
    summaries = []
    for chunk in split_text_into_chunks(text):
        try:
            summary_chunk = summarizer(
                chunk,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary_chunk)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {e}]")
    return " ".join(summaries)

# ==================== Q/A ====================
def answer_question(question, text):
    sentences = text.split(". ")
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, sentence_embeddings, top_k=5)[0]
    answers = [sentences[hit['corpus_id']] for hit in hits if hit['score'] > 0.3]
    return " ".join(answers) if answers else "No relevant answer found."

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        .home-background {
            background: linear-gradient(135deg, #1e1e2f, #2c2c54, #4b3869);
            padding: 40px;
            border-radius: 12px;
            color: white;
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            background: -webkit-linear-gradient(#f8d66d, #ffcc00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            opacity: 0.9;
            font-style: italic;
        }
        .footer {
            text-align: center;
            padding: 15px;
            margin-top: 30px;
            opacity: 0.7;
            font-size: 14px;
            color: #bbb;
        }
        .stButton>button {
            background: linear-gradient(90deg, #b993d6, #8ca6db);
            color: white;
            border-radius: 8px;
            padding: 10px 18px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #8ca6db, #b993d6);
        }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
menu = st.sidebar.radio(
    "📑 Menu",
    ["🏠 Home", "📂 Upload & Preview", "📝 Summarize", "💬 Ask a Question", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload a clear, text-based PDF for the best results.")
st.sidebar.markdown("**Version:** 2.0 🚀")

# ==================== HOME PAGE ====================
if menu == "🏠 Home":
    with st.container():
        st.markdown("<div class='home-background'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>StudyMate!</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Your AI-powered PDF Summarizer & Q/A Assistant</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #ffcc00;'>", unsafe_allow_html=True)
        st.write("Welcome to **StudyMate**, where you can upload large PDFs, get AI-generated summaries, and ask intelligent questions based on the content.")
        st.write("Use the **menu** on the left to explore features.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== UPLOAD PAGE ====================
elif menu == "📂 Upload & Preview":
    uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.markdown("### 📄 PDF Text Preview")
        st.info(pdf_text[:1200] + "..." if len(pdf_text) > 1200 else pdf_text)
        st.session_state["pdf_text"] = pdf_text

# ==================== SUMMARIZE PAGE ====================
elif menu == "📝 Summarize":
    if "pdf_text" in st.session_state:
        if st.button("📝 Generate Summary"):
            with st.spinner("Summarizing large document... Please wait..."):
                summary = summarize_large_text(st.session_state["pdf_text"])
            st.markdown("### 📝 Summary")
            st.success(summary)
            st.session_state["summary"] = summary
    else:
        st.warning("⚠ Please upload a PDF first from the 'Upload & Preview' section.")

# ==================== ASK QUESTION PAGE ====================
elif menu == "💬 Ask a Question":
    if "pdf_text" in st.session_state:
        user_question = st.text_input("💬 Type your question here:")
        if st.button("🔍 Get Answer"):
            with st.spinner("Searching for the best answer..."):
                answer = answer_question(user_question, st.session_state["pdf_text"])
            st.markdown("### 📢 Answer")
            st.success(answer)
    else:
        st.warning("⚠ Please upload a PDF first from the 'Upload & Preview' section.")

# ==================== ABOUT PAGE ====================
elif menu == "ℹ️ About":
    st.markdown("## ℹ️ About StudyMate")
    st.write("""
    StudyMate is an AI-powered tool that:
    - 📂 Extracts text from PDFs
    - 📝 Summarizes large documents
    - 💬 Answers questions based on PDF content
    
    Built using **Streamlit**, **Hugging Face Transformers**, and **Sentence Transformers**.
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<div class='footer'>👑 Built by Code Tech Titans | StudyMate 2025</div>", unsafe_allow_html=True)
