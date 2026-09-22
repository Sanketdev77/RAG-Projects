# This file contains the common configuration used across the project.
# It defines project paths, storage locations, and the embedding model.


from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent


# Input documents
DATA_DIR = BASE_DIR / "documents"


# FAISS vector store
STORE_DIR = BASE_DIR / "store"

INDEX_PATH = STORE_DIR / "faiss_index"
CHUNKS_PATH = STORE_DIR / "chunks.pkl"


# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"