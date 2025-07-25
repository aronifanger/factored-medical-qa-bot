import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RUN = "run_1"
USE_SUBSET = True
SUBSET_SIZE = 1000

# Check if the run directory exists
if not os.path.exists(os.path.join(ROOT_DIR, 'data', RUN)):
    os.makedirs(os.path.join(ROOT_DIR, 'data', RUN))
if not os.path.exists(os.path.join(ROOT_DIR, 'models', RUN)):
    os.makedirs(os.path.join(ROOT_DIR, 'models', RUN))

DATA_SOURCE = "https://drive.google.com/file/d/1vXyLOFRc98f097x4CrK9gOOb3JsKvPmN/view?usp=sharing"
DATASET_PATH = os.path.join(ROOT_DIR, 'data', RUN, 'intern_screening_dataset.csv')
KB_CHUNKS_PATH = os.path.join(ROOT_DIR, 'data', RUN, 'knowledge_base_chunks.csv')

TRAIN_DATA_PATH = os.path.join(ROOT_DIR, 'data', RUN, 'squad_train_data.json')
VAL_DATA_PATH = os.path.join(ROOT_DIR, 'data', RUN, 'squad_val_data.json')
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data', RUN, 'squad_test_data.json')

FAISS_INDEX_PATH = os.path.join(ROOT_DIR, "models", RUN, "faiss_index.bin")
FAISS_PATH = os.path.join(ROOT_DIR, "models", RUN, "faiss_index.bin")
METADATA_PATH = os.path.join(ROOT_DIR, "models", RUN, "chunks_metadata.pkl")
MODEL_PATH = os.path.join(ROOT_DIR, "models", RUN, "final_model")
TRAINING_SUMMARY_PATH = os.path.join(ROOT_DIR, "models", RUN, "training_summary.json")

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_CHECKPOINT = 'distilbert-base-cased-distilled-squad'
