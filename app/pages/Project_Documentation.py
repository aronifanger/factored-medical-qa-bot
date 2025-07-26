import streamlit as st
import pandas as pd

st.set_page_config(page_title="Project Documentation", page_icon="📄", layout="wide")

st.title("📄 Project Documentation")

st.markdown("""
This page documents the technical approach, performance, and architecture of the Medical QA Bot.
""")

st.header("1. Solution Overview")
st.markdown("""
To address the challenge, we implemented an **Extractive Question Answering (EQA)** system. This approach combines a dense retrieval system with a reader model to find and extract answers from a knowledge base.

The pipeline works in two main stages:
1.  **Retriever**: A vector database built with **FAISS** is used to retrieve the most relevant text chunks from the knowledge base in response to a user's question. We use the `all-MiniLM-L6-v2` model from `sentence-transformers` to generate the text embeddings.
2.  **Reader**: A **BERT-based** model (`distilbert-base-cased-distilled-squad`), fine-tuned on the provided dataset, scans the retrieved chunks to identify and extract the precise span of text that answers the question.

#### Rationale
This architecture was chosen because:
- The challenge explicitly required training a custom model, making off-the-shelf LLM APIs unsuitable.
- The use of LLMs was explicitly forbidden.
- The Retriever-Reader model was a state-of-the-art approach for open-domain QA before the widespread adoption of large-scale generative models, making it a robust and appropriate choice.
""")

st.header("2. Performance Evaluation")
st.markdown("""
The system's performance was evaluated in two parts: the retriever's ability to find the correct documents and the reader's ability to extract the correct answer.
""")

st.subheader("Retriever Performance (Recall@3)")
st.markdown("""
The retriever was evaluated on its ability to find the document containing the correct answer within its top 3 retrieved results.
""")
retriever_data = {
    'Metric': ['Recall@3', 'Total Questions', 'Hits (Correct Doc Found)', 'Misses (Correct Doc Not Found)'],
    'Score': ['72.60%', '500', '363', '137']
}
st.table(pd.DataFrame(retriever_data).set_index('Metric'))


st.subheader("Question-Answering Performance (F1-Score)")
st.markdown("""
The end-to-end model was evaluated using Exact Match (EM) and F1-Score, which measures the overlap between the predicted and ground-truth answers.
""")
qa_data = {
    "Method": ["Direct QA (BERT only)", "Retrieval + QA (Full Pipeline)"],
    "Dataset": ["Test", "Test"],
    "Exact Match": ["60.00%", "0.00%"],
    "F1 Score": ["81.75%", "32.72%"]
}
st.table(pd.DataFrame(qa_data))
st.markdown("""
**Analysis:**
- The fine-tuned BERT model performs well when given the correct context directly (**81.75% F1**).
- The performance drops significantly when combined with the retriever (**32.72% F1**). This suggests that while the retriever finds the correct document often, the context formed by combining multiple chunks may be too noisy for the reader.
""")


st.header("3. Limitations")
st.markdown("""
1.  **Dataset Formatting**: The provided dataset was not structured for Extractive Question Answering, as it lacks distinct `context` and `answer` fields where the answer is a direct span of the context. This introduced a bias where the model often extracts the entire text chunk instead of a more concise answer.
2.  **Time Constraints**: The project timeline was limited, which restricted the opportunity for extensive iteration, experimentation, and refinement of the models and code.
""")

st.header("4. Potential Improvements")
st.markdown("""
-   **Model Performance**: The most significant improvement would come from correctly formatting the training data. By creating explicit `(question, context, answer_span)` triplets, the model could be trained to extract more precise answers.
-   **Retriever Performance**: More time could be dedicated to experimenting with different embedding models and chunking strategies to improve the `Recall@k` score.
-   **Test-Driven Development (TDD)**: Given the time constraints, TDD was not adopted. For a long-term project, incorporating tests from the beginning would improve code quality and reliability.
-   **Cross-Platform Compatibility**: The project was developed on Windows. For broader compatibility and easier deployment, the environment should be containerized using Docker with a Linux base.
-   **Code Review and Refinement**: A thorough code review would help enforce consistent coding standards, improve function-level documentation, and refactor modules for better clarity.
""") 