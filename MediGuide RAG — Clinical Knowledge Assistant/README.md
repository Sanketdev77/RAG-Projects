# 🏥 MediGuide RAG — Clinical Knowledge Assistant

MediGuide RAG is a healthcare-focused Retrieval-Augmented Generation (RAG) application that allows users to ask questions from trusted clinical documents.

The application retrieves relevant information from healthcare PDF documents, provides the retrieved context to an LLM, and generates a grounded response with source references.

> ⚕️ This project is designed for clinical knowledge retrieval and is not intended for diagnosis or personalized medical treatment.

---

## 🚀 Project Overview

The project demonstrates an end-to-end RAG pipeline using:

- Python
- LangChain
- Sentence Transformers
- FAISS
- Groq LLM
- Streamlit
- PDF document processing

The current knowledge base contains the WHO HEARTS-D guideline related to Type 2 Diabetes.

---

## 🧠 RAG Architecture

The project follows the standard Retrieval-Augmented Generation architecture:

<img width="1004" height="540" alt="image" src="https://github.com/user-attachments/assets/cce1f008-5490-428a-9557-afe66a414fc7" />


### RAG Flow

```text
Healthcare PDF Documents
          ↓
    Document Loading
          ↓
       Chunking
          ↓
     Embeddings
          ↓
   FAISS Vector Store
          ↓
      User Query
          ↓
    Query Embedding
          ↓
   Similarity Search
          ↓
   Relevant Chunks
          ↓
  Context + User Query
          ↓
       Groq LLM
          ↓
   Grounded Response
          ↓
    Sources / Pages

🔄 How the RAG Pipeline Works
1. Document Loading

Healthcare PDF documents are loaded using PyPDFLoader.

2. Chunking

Large documents are divided into smaller overlapping chunks using RecursiveCharacterTextSplitter.

3. Embedding Generation

Each document chunk is converted into a numerical vector using the all-MiniLM-L6-v2 Sentence Transformer model.

4. Vector Storage

The generated embeddings are stored in a FAISS vector index for efficient similarity search.

5. Query Embedding

When a user asks a question, the question is converted into an embedding using the same embedding model.

6. Similarity Search

FAISS compares the query embedding with document embeddings and retrieves the most relevant chunks.

7. Context Building

The retrieved chunks are combined with the user's question to create the context for the LLM.

8. LLM Generation

The Groq LLM generates an answer using the retrieved context.

9. Source References

The application displays the source document and page information associated with the retrieved content.

📁 Project Structure
MediGuideRAG/
│
├── assets/
│   └── healthcare_bg.jpg
│
├── documents/
│   └── WHO_HEARTS_D_Diabetes.pdf
│
├── ingestion/
│   ├── document_loader.py
│   └── vector_store.py
│
├── retrieval/
│   └── retriever.py
│
├── generation/
│   └── rag_chain.py
│
├── store/
│   ├── faiss_index
│   └── chunks.pkl
│
├── app.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md

📌 File Responsibilities
File	           Responsibility
config.py	          Common project paths and configuration
document_loader.py	Loads PDFs and creates chunks
vector_store.py	Creates embeddings and stores them in FAISS
retriever.py	Retrieves relevant chunks for a user query
rag_chain.py	Combines retrieved context with the query and generates the answer
app.py	          Streamlit user interface
documents/	Healthcare PDF knowledge base
store/	          FAISS index and stored document chunks
.env	          API keys and environment configuration

🛠️ Technologies Used
> Python
> LangChain
> LangChain Community
> Sentence Transformers
> FAISS
> Groq
> Streamlit
> PyPDF
> python-dotenv

⚙️ Setup
1. Clone the repository
git clone <your-repository-url>
cd MediGuideRAG

2. Create the environment
uv venv
Activate the environment or use uv run directly.

3. Install dependencies
uv pip install -r requirements.txt

4. Configure environment variables

Create a .env file:
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=openai/gpt-oss-120b

Do not commit the .env file to GitHub.

▶️ Run the Project
Step 1 — Build the Vector Store

Run the ingestion pipeline:
uv run python -m ingestion.vector_store

This performs:

PDF
 ↓
Loading
 ↓
Chunking
 ↓
Embedding Generation
 ↓
FAISS Index Creation

It generates:

store/
├── faiss_index
└── chunks.pkl

Step 2 — Start Streamlit
uv run streamlit run app.py

Open the local Streamlit URL displayed in the terminal.

💬 Example Questions

Try questions such as:

Q. What are the risk factors for Type 2 diabetes?
Q. How is Type 2 diabetes diagnosed?
Q. What lifestyle interventions are recommended for diabetes?

The application retrieves relevant information from the clinical knowledge base and generates a grounded response.

🔐 Grounded Response

The LLM is instructed to answer using only the retrieved document context.

If the required information is not available in the knowledge base, the system responds:

I don't know based on the provided documents.

This helps reduce unsupported or hallucinated responses.

🎯 Key Features
📄 PDF document ingestion
✂️ Document chunking
🧠 Semantic embeddings
🔎 FAISS similarity search
🤖 LLM-powered response generation
📚 Source and page references
🏥 Healthcare-focused knowledge base
💻 Streamlit web interface
🔐 Environment-based API key management
🛡️ Grounded responses using retrieved context
