import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# =========================
# LLM CONFIG (LOCAL MODEL)
# =========================
MODEL_NAME = "google/flan-t5-base"   # local model, no API call

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_llm():
    """Load tokenizer + model once and cache."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    return tokenizer, model


def generate_answer(question: str, context: str) -> str:
    """Use local FLAN-T5 to generate answer from context + question."""
    tokenizer, model = load_llm()

    prompt = f"""You are a helpful assistant.

Use the following context from a PDF to answer the question.

Context:
{context}

Question: {question}

Answer clearly and briefly:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.5,
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.header("📄 AI PDF Chatbot (Local FLAN-T5 Version)")

with st.sidebar:
    st.title("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None:
    # 1. Read PDF text
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text

    if not full_text.strip():
        st.error("Could not extract any text from this PDF.")
    else:
        # 2. Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_text(full_text)

        # 3. Create embeddings + vector store
        with st.spinner("Processing PDF..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_store = FAISS.from_texts(chunks, embeddings)

        st.success("✅ PDF processed successfully!")

        # 4. Ask question
        question = st.text_input("Ask a question about your document:")

        if question:
            # 5. Retrieve relevant chunks
            docs = vector_store.similarity_search(question, k=4)
            if not docs:
                st.error("No relevant text found in the PDF for this question.")
            else:
                context = "\n\n".join(d.page_content for d in docs)

                # 6. Use local FLAN-T5 to answer
                with st.spinner("Thinking... (local model)"):
                    answer = generate_answer(question, context)

                st.write("### ✅ Answer")
                st.write(answer)
