You work as an AI developer in "Central railways".
Your task is to create a simple chatbot which will be used by all the customers to answer any questions regarding "REFUND POLICY"

1. App
    ingestion.py  - PDF, chunk, embedd, vectorDB(faiss)
    retriever.py  - question, embedded, llm, retrieve the relavant chunks from the vectorDB
    rag_chain.py  - retrived chunks , llm , answer
    agents.py     - orchestration layer

2. Data
    - all the input PDF's

3. Store
    - all the generated chunks and vector Embeddings are stored

4. main.py
    - CLI interfaces

Working:
1. PDF documents
2. Document loader
3. Langchain documents
4. Chunking -> Text chunks
5. Embedding model -> Vectors -> FAISS vectorDB
6. User query -> Query embedding -> similarity search on VectorDB
7. Retrieve top-K chunks
8. LLM (Groq)
9. Answer