import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical QA Bot",
    page_icon="🤖",
    layout="centered"
)

# --- Title and Description ---
st.title("🤖 Medical QA Bot")
st.markdown("""
Welcome to the Medical QA Bot! This bot is trained to answer questions about medical diseases.
**How to use:**
1.  Type your question in the text box below.
2.  Click the "Ask" button.
3.  The answer and the context used to generate it will be displayed.
""")

# --- API Address ---
API_URL = "http://127.0.0.1:8000/ask"

# --- User Interaction ---
question = st.text_input("What is your medical question?", "")

if st.button("Ask"):
    if question:
        try:
            # --- API Call ---
            with st.spinner("Finding the answer..."):
                response = requests.post(API_URL, json={"question": question})
                response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)

                data = response.json()

                # --- Display the Answer ---
                st.subheader("Answer:")
                st.success(data['answer'])

                with st.expander("View Context Used"):
                    for i, context in enumerate(data['context']):
                        st.markdown(f"### Context {i+1}")
                        st.write(context.replace("---", "\n\n").replace("#", ""))
                        st.write("-"*100 + "\n")
                
                st.write(f"**Confidence Score:** {data['score']:.4f}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a question.")

# --- Footer ---
st.markdown("---")
st.markdown("Medical Assistant Bot Assignment - coding challenge.") 