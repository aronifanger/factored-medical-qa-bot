import pandas as pd
import random

from tqdm import tqdm   
from transformers import AutoTokenizer


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
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Creates SQuAD-like datasets (train, validation, test) from a DataFrame.
    For each sample, the context is the answer itself.
    """
    
    squad_data = []
    
    print("Generating SQuAD-style data with context as the answer...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(row['answer']):
            continue
        
        question = row['question']
        answer_text = row['answer'].strip()
        
        # The context is the answer itself.
        context = answer_text
        answer_start = 0

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
