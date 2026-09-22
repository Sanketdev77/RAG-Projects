# This file finds the most relevant document chunks for a user's question.
# Flow: User Query -> Query Embedding -> Similarity Search -> Relevant Chunks

import pickle

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ---------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------

def retriever(query: str, k: int = 5):
    """
    Retrieve the top-k document chunks relevant to
    the user's query.

    Parameters
    ----------
    query : str
        Natural language question from the user.

    k : int, default=5
        Number of relevant chunks to retrieve.

    Returns
    -------
    list
        List of retrieved LangChain Document objects.
    """

    # -----------------------------------------------------
    # STEP 1: VALIDATE VECTOR STORE
    # -----------------------------------------------------

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. "
            "Please run the ingestion pipeline first."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Document chunks not found. "
            "Please run the ingestion pipeline first."
        )

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    # -----------------------------------------------------
    # STEP 2: LOAD STORED CHUNKS
    # -----------------------------------------------------

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    # -----------------------------------------------------
    # STEP 3: LOAD FAISS INDEX
    # -----------------------------------------------------

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    # -----------------------------------------------------
    # STEP 4: CREATE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    # -----------------------------------------------------
    # STEP 5: SEARCH FAISS
    # -----------------------------------------------------

    # Make sure k does not exceed the number
    # of indexed documents.
    k = min(k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    # -----------------------------------------------------
    # STEP 6: MAP VECTOR IDs BACK TO DOCUMENT CHUNKS
    # -----------------------------------------------------

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        # FAISS can return -1 when no result exists
        if index_id == -1:
            continue

        chunk = chunks[index_id]

        # Store similarity score in metadata
        chunk.metadata["similarity_score"] = float(score)

        results.append(chunk)

    return results