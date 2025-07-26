import faiss
import pickle
import torch

from config import FAISS_INDEX_PATH, METADATA_PATH, MODEL_PATH, EMBEDDING_MODEL_NAME
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Medical QA Bot API",
    description="API for answering medical questions using a fine-tuned model and a vector database.",
    version="1.0.0"
)

# --- Data Models ---
class Query(BaseModel):
    question: str
    top_k: int = 3  # Number of contexts to retrieve

class Answer(BaseModel):
    question: str
    answer: str
    context: list[str]
    score: float

# --- Global Variables ---
# These will be loaded once at startup
retriever = None
qa_pipeline = None

# --- Startup Logic ---
@app.on_event("startup")
def load_models():
    """Load all necessary models and data into memory when the API starts."""
    global retriever, qa_pipeline
    
    print("--- Loading models and data... ---")

    # 1. Load FAISS index and metadata for context retrieval
    try:
        print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)
        
        print(f"Loading metadata from: {METADATA_PATH}")
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        retriever = {
            "index": index,
            "metadata": metadata,
            "embedding": embedding_model
        }
        print(f"FAISS index and metadata loaded successfully. Index contains {index.ntotal} vectors.")

    except FileNotFoundError:
        raise RuntimeError(f"Could not find FAISS index or metadata at the specified paths. Please run `prepare_vector_database.py` first.")
    except Exception as e:
        raise RuntimeError(f"Error loading retriever components: {e}")

    # 2. Load the fine-tuned Question Answering model and tokenizer
    try:
        print(f"Loading QA model from: {MODEL_PATH}")
        device = 0 if torch.cuda.is_available() else -1
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print(f"QA pipeline loaded successfully on device: {'cuda' if device == 0 else 'cpu'}")

    except Exception as e:
        raise RuntimeError(f"Error loading the QA model from '{MODEL_PATH}': {e}")
    
    print("--- Models loaded and API is ready! ---")
    
# --- API Endpoints ---
@app.get("/", summary="Health Check", description="Check if the API is running.")
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/ask", response_model=Answer, summary="Ask a Medical Question")
def ask_question(query: Query):
    """
    Receives a question, finds relevant context from the vector database,
    and returns the answer predicted by the QA model.
    """
    if not retriever or not qa_pipeline:
        raise RuntimeError("Models are not loaded. Please check the startup process.")

    # 1. Find relevant context using FAISS
    question_embedding = retriever["embedding"].encode([query.question], convert_to_numpy=True)
    _, indices = retriever["index"].search(question_embedding, k=query.top_k)
    context_list = [retriever["metadata"][i]['answer_chunk'] for i in indices[0]]
    context = " ".join(context_list)

    # 2. Use the QA pipeline to get the answer
    result = qa_pipeline(question=query.question, context=context, max_answer_len=512)

    return Answer(
        question=query.question,
        answer=result["answer"],
        context=context_list,
        score=result["score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True) 