from tqdm import tqdm   
from transformers import AutoTokenizer
import json
import nltk
import pandas as pd

def chunk_text_by_sentence(text: str, tokenizer, chunk_size: int = 400, overlap_sentences: int = 1) -> list[str]:
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

def create_squad_dataset_pipeline(df_original: pd.DataFrame, tokenizer, max_answer_len_tokens: int = 512, 
                                 train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Complete pipeline to create the knowledge base (chunks) and SQuAD training data.
    Split into train/validation/test datasets.
    
    Args:
        df_original: Original DataFrame with questions and answers
        tokenizer: Tokenizer for text processing
        max_answer_len_tokens: Maximum length for answers in tokens
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
    
    Returns:
        tuple: (unique_chunks, train_data, val_data, test_data)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # 1. Create the knowledge base (KB) with the new chunking function
    print("Step 1: Aggregating answers to create long contexts...")
    df_contexts = df_original.groupby('question')['answer'].apply(lambda x: ' '.join(str(i) for i in x)).reset_index()
    df_contexts.columns = ['topic', 'long_context']

    print("Step 2: Applying sentence-based chunking to long contexts...")
    all_chunks = []
    for context in tqdm(df_contexts['long_context']):
        chunks = chunk_text_by_sentence(context, tokenizer, chunk_size=400, overlap_sentences=1)
        all_chunks.extend(chunks)

    unique_chunks = list(set(all_chunks))
    print(f"Knowledge base created with {len(unique_chunks)} unique chunks.")

    # 2. Create the SQuAD training dataset
    all_training_data = []
    print(f"\nStep 3: Generating training data from {len(df_original)} original rows...")
    
    # Counter for debugging
    found_in_chunk_count = 0

    for _, row in tqdm(df_original.iterrows(), total=df_original.shape[0]):
        question = str(row['question'])
        answer_text = str(row['answer'])
        answer_tokens_len = len(tokenizer.tokenize(answer_text))

        if 0 < answer_tokens_len < max_answer_len_tokens:
            # --- SCENARIO A: Short Answer ---
            for chunk_context in unique_chunks:
                # The 'in' search is now much more likely to work
                if answer_text in chunk_context:
                    start_index = chunk_context.find(answer_text)
                    all_training_data.append({
                        'context': chunk_context, 'question': question,
                        'answers': {'text': [answer_text], 'answer_start': [start_index]}
                    })
                    found_in_chunk_count += 1
                    break
        elif answer_tokens_len >= max_answer_len_tokens:
            # --- SCENARIO B: Long Answer ---
            long_answer_as_context = answer_text
            answer_spans = chunk_text_by_sentence(long_answer_as_context, tokenizer, chunk_size=150, overlap_sentences=1)
            for span in answer_spans:
                start_index = long_answer_as_context.find(span)
                if start_index != -1:
                    all_training_data.append({
                        'context': long_answer_as_context, 'question': question,
                        'answers': {'text': [span], 'answer_start': [start_index]}
                    })

    print(f"\nStep 4: Splitting data into train/validation/test sets...")
    
    # Shuffle the data for random splits
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(all_training_data)
    
    # Calculate split indices
    total_examples = len(all_training_data)
    train_end = int(total_examples * train_ratio)
    val_end = int(total_examples * (train_ratio + val_ratio))
    
    # Split the data
    train_data = all_training_data[:train_end]
    val_data = all_training_data[train_end:val_end]
    test_data = all_training_data[val_end:]
    
    print(f"\nDataset split completed:")
    print(f"  Total examples: {total_examples}")
    print(f"  Training set: {len(train_data)} examples ({len(train_data)/total_examples:.1%})")
    print(f"  Validation set: {len(val_data)} examples ({len(val_data)/total_examples:.1%})")
    print(f"  Test set: {len(test_data)} examples ({len(test_data)/total_examples:.1%})")
    print(f"  For Scenario A (short answers), {found_in_chunk_count} answers were found in chunks.")
    
    return unique_chunks, train_data, val_data, test_data
