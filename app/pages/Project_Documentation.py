import streamlit as st

st.set_page_config(page_title="Project Documentation", page_icon="📄", layout="wide")

st.title("📄 Project Documentation")

st.markdown("""
This section documents the approach used to develop the Medical QA Bot, as requested in the challenge.
""")

st.header("1. Assumptions Made")
st.markdown("""
- **API Availability:** The Streamlit interface assumes that the FastAPI API (`src/api.py`) is running and accessible at `http://127.0.0.1:8000`.
- **Context Quality:** The quality of the bot's response depends directly on the relevance of the text chunks retrieved by `FAISS`. It is assumed that the found contexts are sufficient to formulate a correct answer.
- **Execution Environment:** The project was developed and tested in a Python 3.10+ environment with the dependencies listed in `pyproject.toml`.
- **Model Generalization:** The model was trained on a specific dataset. Its ability to answer questions outside this medical domain is limited.
""")

st.header("2. Model Performance")
st.markdown("""
A formal performance evaluation is detailed in the execution reports, but we can highlight some qualitative points:

**Strengths:**
- **Fast Retrieval:** Using `FAISS` for nearest neighbor search allows for extremely fast context retrieval, even with a large vector database.
- **Direct Answers:** The Question Answering (QA) model is effective at extracting the exact answer from a context text, resulting in concise answers.
- **Scalability:** The architecture with an API decoupled from the frontend allows both to be scaled independently.

**Weaknesses:**
- **Context Dependency:** If the retrieved context does not contain the answer, the model cannot answer or may provide an incorrect answer based on partial information.
- **Sensitivity to Question Phrasing:** Questions phrased very differently from the original text can lead to inadequate context retrieval.
- **Does Not Generate New Information:** The model does not "reason" or create new sentences; it only extracts excerpts from the provided context.
""")

st.header("3. Potential Improvements")
st.markdown("""
- **Dataset Augmentation:** Use more medical data, possibly from reliable sources like PubMed, to enrich the vector knowledge base and refine the QA model.
- **Hybrid Retrieval:** Combine vector similarity search (dense) with traditional keyword search (sparse, like TF-IDF or BM25) to improve the relevance of retrieved documents.
- **Generative Model (RAG):** Replace the extractive QA model with a generative language model (like GPT or T5) in a Retrieval-Augmented Generation (RAG) flow. This would allow for more fluid and natural answers, rather than just extracting snippets.
- **User Feedback:** Implement a system where the user can rate the quality of the answer (👍/👎). This feedback could be used to continuously refine the retrieval or QA model.
- **Response Streaming:** For generative models, answers could be streamed word by word to improve the user's perception of speed.
""") 