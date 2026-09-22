# This file provides the Streamlit user interface for the MediGuide RAG application.
# Flow: User Question -> RAG Pipeline -> Answer + Sources

import streamlit as st

from generation.rag_chain import generate_answer


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="MediGuide RAG",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.title("🏥 MediGuide RAG")
st.subheader("Clinical Knowledge Assistant")

st.write(
    """
    Ask questions about the healthcare documents available
    in the knowledge base. The system retrieves relevant
    information from the documents and generates a
    grounded response using an LLM.
    """
)

st.divider()


# ---------------------------------------------------------
# USER QUERY
# ---------------------------------------------------------

query = st.text_input(
    "Enter your healthcare question:",
    placeholder="e.g. What are the risk factors for Type 2 diabetes?"
)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------

if st.button("Ask", type="primary"):

    if not query.strip():
        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the clinical knowledge base..."
        ):

            try:
                result = generate_answer(query)

                answer = result["answer"]
                sources = result["sources"]

                # -----------------------------------------
                # DISPLAY ANSWER
                # -----------------------------------------

                st.subheader("Answer")

                st.write(answer)

                
            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(e)
