import faiss
import nltk
import pandas as pd
import pickle
import time

from config import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    print("Downloading the 'punkt' sentence tokenizer from NLTK...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    print("Downloading the 'punkt_tab' sentence tokenizer from NLTK...")
    nltk.download('punkt_tab')


def chunk_text(text: str, tokenizer, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """
    Divide a long text into smaller chunks, respecting the sentence boundaries
    and with a defined overlap in number of tokens.

    Args:
        text (str): The complete text to be divided.
        tokenizer: The tokenizer from the Transformers library to be used to count the tokens.
        chunk_size (int): The maximum size of each chunk in tokens.
        chunk_overlap (int): The number of tokens of overlap between consecutive chunks.

    Returns:
        list[str]: A list of strings, where each string is a chunk of text.
    """
    if not text:
        return []

    # 1. Dividir o texto em sentenças usando NLTK
    sentences = nltk.sent_tokenize(text)

    # 2. Tokenizar todas as sentenças e guardar seus tokens
    tokens = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    
    chunks = []
    current_chunk_tokens = []
    
    for i, sentence_tokens in enumerate(tokens):
        # Se adicionar a próxima sentença ultrapassar o tamanho do chunk
        if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size:
            # Finaliza o chunk atual e adiciona à lista
            if current_chunk_tokens:
                chunk_str = tokenizer.decode(current_chunk_tokens)
                chunks.append(chunk_str.strip())
            
            # 3. Começa um novo chunk com sobreposição
            # Pega os últimos tokens do chunk que acabamos de criar para formar a sobreposição
            overlap_tokens = current_chunk_tokens[-chunk_overlap:] if chunk_overlap > 0 and current_chunk_tokens else []
            current_chunk_tokens = overlap_tokens + sentence_tokens
        else:
            current_chunk_tokens.extend(sentence_tokens)

    # Adiciona o último chunk que sobrou
    if current_chunk_tokens:
        chunk_str = tokenizer.decode(current_chunk_tokens)
        chunks.append(chunk_str.strip())
        
    return chunks


def generate_chunks(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    df_clean = df.reset_index(drop=True)
    df_clean["original_index"] = df_clean.index
    df_clean = df_clean.dropna(subset=['answer'])

    # For each question, concatenate all answers and collect the list of original indices
    df_concat = df_clean.groupby('question').agg({
        'answer': lambda x: '\n---\n'.join(x),
        'original_index': lambda x: list(x)
    }).reset_index()

    # Create a new column 'question_index' with the index of each unique question
    df_concat['question_index'] = df_concat.index

    # Group by 'answer' to aggregate duplicated answers
    df_answer_grouped = df_concat.groupby('answer').agg({
        # Concatenate all unique questions into a single string, separated by '\n---\n'
        'question': lambda x: '\n---\n'.join(sorted(set(x))),
        # Concatenate all lists of original_index into a single list
        'original_index': lambda x: sum(x, []),
        # Collect all question_index values into a list
        'question_index': lambda x: list(x)
    }).reset_index()

    # Create a new column 'answer_index' as a unique index for each answer
    df_answer_grouped['answer_index'] = df_answer_grouped.index

    # Use the chunk_text function to split the 'answer' column into 'answer_chunk'
    df_answer_grouped['answer_chunk'] = df_answer_grouped['answer'].apply(
        lambda x: chunk_text(x, tokenizer)
    )

    return df_answer_grouped.explode('answer_chunk')

def create_and_save_faiss_index(
    metadata: list[dict],
    model_name: str,
    index_path: str,
    metadata_path: str
):

    print(f"Loading Sentence Transformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(metadata)} chunks... This may take a few minutes.")
    start_time = time.time()
    # Generate embeddings using the text field (ex: 'answer_chunk')
    embeddings = model.encode([m['answer_chunk'] for m in metadata], show_progress_bar=True, convert_to_numpy=True)
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

    d = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {d}...")
    index = faiss.IndexFlatL2(d)
    print("Adding vectors to the FAISS index...")
    index.add(embeddings)
    print(f"The index now contains {index.ntotal} vectors.")

    print(f"Saving FAISS index to '{index_path}'...")
    faiss.write_index(index, index_path)

    print(f"Saving metadata (chunks) to '{metadata_path}'...")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\nProcess completed successfully!")
    print(f"Index saved at: {index_path}")
    print(f"Metadata saved at: {metadata_path}")