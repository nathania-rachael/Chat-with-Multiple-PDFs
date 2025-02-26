import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from dotenv import load_dotenv

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))  
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    """Split extracted text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Convert text chunks into embeddings and store in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a conversational chain using Gemini AI."""
    prompt_template = """Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided content, just say, "Answer is not available in the context." 
    Do not provide incorrect information.

    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Handle user questions and retrieve answers from stored embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs.")
            return
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing query: {e}")


def main():
    """Streamlit UI setup for Chat with PDFs."""
    st.set_page_config(page_title="Chat With Multiple PDFs")
    st.header("Chat with PDFs 📄💬")

    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📌 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF processing completed! You can now ask questions.")
                else:
                    st.error("No readable text found in the uploaded PDFs.")

if __name__ == "__main__":
    main()
