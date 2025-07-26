import streamlit as st
import requests
import os

st.set_page_config(page_title="Usage Examples", page_icon="🧪", layout="centered")

st.title("🧪 Usage Examples")

st.markdown("""
Here are some example questions you can ask the bot. 
Click any of the buttons to send the question to the API and see the response.
""")

# --- API Address ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/ask")

# --- Example Questions ---
example_questions = [
    "What is Glaucoma?",
    "What are the symptoms of a stroke?",
    "How can I prevent high blood pressure?",
]

def ask_question(question):
    """Function to call the API and display the response."""
    try:
        with st.spinner(f"Asking: *{question}*"):
            response = requests.post(API_URL, json={"question": question})
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer:")
            st.success(data['answer'])

            with st.expander("View Context Used"):
                for i, context in enumerate(data['context']):
                    st.markdown(f"### Context {i+1}")
                    st.write(context.replace("---", "\n\n").replace("#", ""))
                    st.write("-"*100 + "\n")
            
            st.write(f"**Confidence Score:** {data['score']:.4f}")
            st.markdown("---")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Create Buttons for each Example ---
for q in example_questions:
    if st.button(q, key=q):
        ask_question(q) 