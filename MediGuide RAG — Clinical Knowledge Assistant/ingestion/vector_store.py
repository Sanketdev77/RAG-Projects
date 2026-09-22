# This file converts document chunks into embeddings and stores them in FAISS.
# Flow: Chunks -> Embeddings -> FAISS Vector Store

import pickle

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    STORE_DIR,
    INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME
)

from ingestion.document_loader import (
    load_docs,
    split_docs
)


# ---------------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# ---------------------------------------------------------

def create_embeddings(chunks):
    """
    Generate embeddings for document chunks.

    Parameters
    ----------
    chunks : list
        List of LangChain Document objects.

    Returns
    -------
    embeddings : numpy.ndarray
        Generated document embeddings.
    """

    if not chunks:
        raise ValueError(
            "No chunks provided for embedding."
        )

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(
        f"Generated embeddings with shape: "
        f"{embeddings.shape}"
    )

    return embeddings


# ---------------------------------------------------------
# STEP 4: STORE EMBEDDINGS IN FAISS
# ---------------------------------------------------------

def store_faiss(embeddings, chunks):
    """
    Store document embeddings in FAISS and
    persist the document chunks separately.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Document embeddings.

    chunks : list
        Corresponding LangChain Document objects.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            "No embeddings available for FAISS."
        )

    if not chunks:
        raise ValueError(
            "No chunks available to store."
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings must match "
            "number of chunks."
        )

    # Create storage directory
    STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Determine embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to FAISS
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # Save chunks
    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print(
        f"FAISS index saved to: {INDEX_PATH}"
    )

    print(
        f"Chunks saved to: {CHUNKS_PATH}"
    )


# ---------------------------------------------------------
# STEP 5: MAIN INGESTION PIPELINE
# ---------------------------------------------------------

def build_vector_store():
    """
    Complete document ingestion pipeline:

    PDF
      ↓
    Load documents
      ↓
    Split into chunks
      ↓
    Generate embeddings
      ↓
    Store embeddings in FAISS
    """

    # 1. Load documents
    docs = load_docs(DATA_DIR)

    print(
        f"Loaded {len(docs)} document pages."
    )

    # 2. Split documents into chunks
    chunks = split_docs(docs)

    print(
        f"Split into {len(chunks)} chunks."
    )

    # 3. Generate embeddings
    embeddings = create_embeddings(chunks)

    # 4. Store vectors and chunks
    store_faiss(
        embeddings,
        chunks
    )

    print(
        f"\nIngestion completed successfully.\n"
        f"Processed {len(docs)} pages into "
        f"{len(chunks)} chunks."
    )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()