import pandas as pd
import random
import re

from tqdm import tqdm   
from transformers import AutoTokenizer


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
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Creates SQuAD-like datasets with context that includes distractor answers
    to make the model more robust.
    """
    
    squad_data = []
    all_answers = df['answer'].dropna().unique().tolist()
    
    print("Generating SQuAD-style data with distractor contexts...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(row['answer']):
            continue
        
        question = row['question']
        correct_answer = row['answer'].strip()
        
        # Select two random distractor answers
        distractors = random.sample(
            [ans for ans in all_answers if ans != correct_answer], 2
        )
        
        # Create a context with the correct answer and distractors
        context_pieces = [correct_answer] + distractors
        random.shuffle(context_pieces)
        context = "\n\n---\n\n".join(context_pieces)
        
        answer_start = find_answer_start(context, correct_answer)

        if answer_start != -1:
            qa_pair = {
                'id': str(abs(hash(context + question + correct_answer))),
                'title': "Medical Information",
                'context': context,
                'question': question,
                'answers': {
                    'text': [correct_answer],
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
