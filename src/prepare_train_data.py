import faiss
import json
import nltk
import numpy as np
import pandas as pd
import pickle
import random
import re

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm   
from transformers import AutoTokenizer


def chunk_text_by_sentence(
        text: str,
        tokenizer,
        chunk_size: int = 400,
        overlap_sentences: int = 1
    ) -> list[str]:
    """
    Splits a text into chunks, with overlap based on whole sentences.
    This is a more robust approach to ensure text integrity.
    """
    if not isinstance(text, str) or not text:
        return []

    # 1. Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # 2. Group sentences into chunks
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        
        # If adding the next sentence exceeds the limit
        if current_chunk_tokens + len(sentence_tokens) > chunk_size and current_chunk_sentences:
            # Finalize the current chunk
            chunks.append(" ".join(current_chunk_sentences))
            
            # Start a new chunk with sentence overlap
            current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
            current_chunk_tokens = len(tokenizer.tokenize(" ".join(current_chunk_sentences)))
        
        current_chunk_sentences.append(sentence)
        current_chunk_tokens += len(sentence_tokens)

    # Add the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def find_answer_start(context, answer):
    """
    Finds the starting index of an answer within a context.
    Returns -1 if the answer is not found.
    """
    match = next(re.finditer(re.escape(answer), context), None)
    if match is None:
        return -1
    return match.start()

def create_squad_dataset_pipeline(
    df: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    chunk_size: int,
    overlap_sentences: int,
    faiss_index_path: str,
    metadata_path: str,
    embedding_model_name: str
) -> tuple[list[str], list[dict], list[dict], list[dict]]:
    """
    Creates SQuAD-like datasets (train, validation, test) from a DataFrame.
    This version augments context by retrieving similar chunks from a FAISS index.
    """
    
    # 1. Load FAISS index, metadata, and embedding model
    print("Loading FAISS index, metadata, and embedding model for context augmentation...")
    try:
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"FAISS index with {faiss_index.ntotal} vectors, and metadata for {len(metadata)} chunks loaded.")
    except Exception as e:
        print(f"Error loading FAISS components: {e}")
        print("Please ensure 'prepare_vector_database.py' has been run successfully.")
        exit()

    squad_data = []
    
    print("Generating SQuAD-style data with augmented context...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(row['answer']):
            continue
        
        question = row['question']
        answer_text = row['answer'].strip()
        
        # Retrieve top_k relevant chunks from FAISS
        question_embedding = embedding_model.encode([question], convert_to_numpy=True)
        distances, indices = faiss_index.search(question_embedding, k=3)
        
        retrieved_chunks = [metadata[i]['answer_chunk'] for i in indices[0]]
        
        context_pieces = [chunk for chunk in retrieved_chunks if answer_text.lower() not in chunk.lower()]
        context_pieces.append(answer_text)
        
        # Shuffle and join
        random.shuffle(context_pieces)
        context = "\n\n".join(context_pieces)
        
        answer_start = find_answer_start(context, answer_text)

        if answer_start != -1:
            qa_pair = {
                'id': str(abs(hash(context + question + answer_text))),
                'title': "Medical Information",
                'context': context,
                'question': question,
                'answers': {
                    'text': [answer_text],
                    'answer_start': [answer_start]
                }
            }
            squad_data.append(qa_pair)
        else:
            raise ValueError(f"Answer start is -1 for question: {question}")

    # Shuffle the dataset before splitting
    random.shuffle(squad_data)
    
    # Split the data
    total_size = len(squad_data)
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    train_data = squad_data[:train_end]
    val_data = squad_data[train_end:val_end]
    test_data = squad_data[val_end:]
    
    return train_data, val_data, test_data
