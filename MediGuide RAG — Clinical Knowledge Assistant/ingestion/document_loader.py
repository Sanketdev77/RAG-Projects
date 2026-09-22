# This file loads PDF documents and splits them into smaller chunks.
# Flow: PDF Documents -> Document Loading -> Chunking

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# ---------------------------------------------------------

def load_docs(data_path: Path):
    """
    Load all PDF documents from the specified directory.

    Parameters
    ----------
    data_path : Path
        Directory containing PDF files.

    Returns
    -------
    list
        List of LangChain Document objects.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {data_path}"
        )

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_path}"
        )

    docs = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()

            # Add document name to metadata
            for doc in pdf_docs:
                doc.metadata["document_name"] = pdf_file.name

            docs.extend(pdf_docs)

            print(
                f"Loaded {len(pdf_docs)} pages from "
                f"{pdf_file.name}"
            )

        except Exception as e:
            print(
                f"Error loading {pdf_file.name}: {e}"
            )

    if not docs:
        raise ValueError(
            "No documents could be loaded from the PDF files."
        )

    print(f"Total pages loaded: {len(docs)}")

    return docs


# ---------------------------------------------------------
# STEP 2: SPLIT DOCUMENTS INTO CHUNKS
# ---------------------------------------------------------

def split_docs(docs):
    """
    Split documents into smaller chunks for embedding
    and vector similarity search.

    Parameters
    ----------
    docs : list
        List of LangChain Document objects.

    Returns
    -------
    list
        List of chunked LangChain Document objects.
    """

    if not docs:
        raise ValueError(
            "No documents were provided for chunking."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_documents(docs)

    # Add unique chunk identifier
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Total chunks created: {len(chunks)}")

    return chunks
