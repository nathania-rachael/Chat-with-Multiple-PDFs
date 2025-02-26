# Chat with Multiple PDFs  

A **Streamlit-based AI-powered chatbot** that allows users to upload multiple PDFs and interact with the content using **Google Generative AI (Gemini)**. The system extracts text from PDFs, converts it into embeddings using FAISS, and enables users to query the documents conversationally.  

## Features  

- Upload multiple PDFs  
- Extract and process text from PDFs  
- Store embeddings using FAISS  
- Ask questions and get responses based on document content  
- Uses **Google Generative AI (Gemini Pro)** for responses  

## Installation  

### Prerequisites  
- Python 3.8+  
- An active **Google API Key** (for **Google Generative AI**)  
- **Conda installed**  

### Steps  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/chat-with-multiple-pdfs.git
   cd chat-with-multiple-pdfs
   ```

2. **Create and activate a Conda environment**  
   ```bash
   conda create --name chatpdf-env python=3.8
   conda activate chatpdf-env
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google API Key**  
   Create a `.env` file in the project directory and add:  
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage  

1. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```

2. **Upload PDFs** using the sidebar  
3. **Ask questions** based on the uploaded documents  
4. **Get accurate answers** extracted from the PDFs  

## Technologies Used  

- **Streamlit** - For building the web app UI  
- **Google Generative AI (Gemini Pro)** - For AI-powered responses  
- **PyPDF2** - For extracting text from PDFs  
- **FAISS** - For storing and retrieving document embeddings  
- **LangChain** - For integrating AI models  

## License  

This project is licensed under the MIT License.  

## Contact 

Nathania Rachael - nathaniarachael@gmail.com
