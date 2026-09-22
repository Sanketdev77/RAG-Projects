# This file generates the final answer using the retrieved document context and LLM.
# Flow: Query + Relevant Chunks -> Prompt -> LLM -> Answer

import os

from dotenv import load_dotenv
from groq import Groq

from retrieval.retriever import retriever


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# INITIALIZE GROQ CLIENT
# ---------------------------------------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "openai/gpt-oss-120b"
)


# ---------------------------------------------------------
# MAIN RAG PIPELINE
# ---------------------------------------------------------

def generate_answer(query: str):
    """
    Generate an answer using retrieved document context
    and the Groq LLM.

    Flow:
        User Query
            ↓
        Retriever
            ↓
        Relevant Chunks
            ↓
        Prompt
            ↓
        Groq LLM
            ↓
        Answer
    """

    # -----------------------------------------------------
    # STEP 1: RETRIEVE RELEVANT DOCUMENTS
    # -----------------------------------------------------

    retrieved_chunks = retriever(query)

    if not retrieved_chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }

    # -----------------------------------------------------
    # STEP 2: BUILD CONTEXT
    # -----------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            chunk.page_content
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # STEP 3: BUILD PROMPT
    # -----------------------------------------------------

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    # -----------------------------------------------------
    # STEP 4: GENERATE ANSWER
    # -----------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """
You are a healthcare knowledge assistant.

Answer the user's question strictly using
ONLY the information provided in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not available in the context,
   say: "I don't know based on the provided documents."
4. Keep the answer clear and concise.
5. Do not provide a medical diagnosis.
6. Do not provide personalized medical treatment advice.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    # -----------------------------------------------------
    # STEP 5: COLLECT SOURCE INFORMATION
    # -----------------------------------------------------

    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "document": chunk.metadata.get(
                "document_name",
                "Unknown"
            ),
            "page": chunk.metadata.get(
                "page",
                "Unknown"
            ),
            "score": chunk.metadata.get(
                "similarity_score"
            )
        })

    return {
        "answer": answer,
        "sources": sources
    }