import streamlit as st
import PyPDF2
from transformers import pipeline
from dotenv import load_dotenv
import os
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Hugging Face API key
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY:
    try:
        login(token=HF_API_KEY)
        st.success("🔑 Hugging Face login successful.")
    except Exception as e:
        st.warning(f"Hugging Face login failed: {e}")
else:
    st.info("No HF_API_KEY found. Using public model without authentication.")

# Initialize models
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=-1,
    framework="pt"
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

st.set_page_config(page_title="MemoMate", page_icon="📚", layout="centered")
st.title("📚 MemoMate (Enhanced Version)")
st.caption("AI-powered PDF Q&A, Summarization & Concept Explainer")

# Extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    # OCR fallback
    if not text.strip():
        images = convert_from_bytes(pdf_bytes)
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
    return " ".join(text.split())

# Split text into chunks
def split_text(text, chunk_size=400, overlap=50):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

# Semantic search
def find_relevant_chunks(question, chunks, top_n=3):
    chunk_embeddings = embedding_model.encode(chunks)
    question_embedding = embedding_model.encode([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

# Upload PDF
uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    pdf_text = extract_text_from_pdf(pdf_bytes)

    if not pdf_text.strip():
        st.error("⚠️ No extractable text found in this PDF.")
    else:
        st.success("✅ PDF uploaded and processed!")

        if st.checkbox("Show extracted PDF text"):
            st.text_area("PDF Text Preview", pdf_text, height=300)

        # PDF Summarization
        if st.button("Summarize PDF"):
            with st.spinner("Generating summary..."):
                summary = summarizer(pdf_text, max_length=300, min_length=80, do_sample=False)[0]['summary_text']
                st.markdown(f"**PDF Summary:** {summary}")
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="pdf_summary.txt",
                    mime="text/plain"
                )

        # Q&A Section
        question = st.text_input("Ask a question about the PDF or a concept:")

        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding the answer..."):
                    chunks = list(split_text(pdf_text, chunk_size=400, overlap=50))
                    relevant_chunks = find_relevant_chunks(question, chunks, top_n=3)

                    final_answers = []
                    for chunk in relevant_chunks:
                        answer = qa_pipeline(question=question, context=chunk)
                        if answer.get("score", 0) > 0.0:
                            final_answers.append(answer["answer"])

                    if not final_answers:
                        st.info("No relevant answer found. Trying concept explanation...")
                        # Use QA pipeline with the whole PDF text as context for explanation
                        try:
                            explanation = qa_pipeline(question=f"Explain the concept: {question}", context=pdf_text)
                            if explanation.get("score", 0) > 0.0:
                                st.markdown(f"**Explanation:** {explanation['answer']}")
                        except:
                            st.info("Could not generate explanation.")
                    else:
                        combined_answer = " ".join(list(dict.fromkeys(final_answers)))  # remove duplicates
                        st.markdown(f"**Answer:** {combined_answer}")
                        st.download_button(
                            label="Download Answer",
                            data=combined_answer,
                            file_name="memo_answer.txt",
                            mime="text/plain"
                        )
