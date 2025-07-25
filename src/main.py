from download_data import get_drive_file_id, download_file_from_google_drive
from prepare_train_data import create_squad_dataset_pipeline
from prepare_vector_database import generate_chunks, create_and_save_faiss_index
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from train import prepare_train_features
from datasets import load_dataset

import json
import os
import pandas as pd
import re
import requests
import torch

def download_data(data_source, dataset_path):
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return
    
    file_id = get_drive_file_id(data_source)
    # Tries to get the file name
    response = requests.get(f"https://drive.google.com/uc?export=download&id={file_id}", stream=True)
    content_disp = response.headers.get('content-disposition')
    if content_disp:
        fname_match = re.findall('filename=\"(.+?)\"', content_disp)
        filename = fname_match[0] if fname_match else f"{file_id}.csv"
    else:
        filename = f"{file_id}.csv"
    destination = os.path.join(dataset_path)
    print(f"Downloading to: {destination}")
    download_file_from_google_drive(file_id, destination)
    print("Download completed!")
    
def prepare_vector_database(dataset_path, embedding_model_name, faiss_index_path, metadata_path, use_subset, subset_size):
    # Load the same tokenizer that you will use in your BERT/DistilBERT model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    # Load the generated chunks
    try:
        if use_subset:
            df_original = pd.read_csv(dataset_path).sample(n=subset_size, random_state=42)
        else:
            df_original = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found.")
        print("Please run `src/download_data.py` first to download the original dataset.")
        exit()

    df_chunks = generate_chunks(df_original, tokenizer)

    # The dataframe is expected to have an 'answer_chunk' column
    # Convert it to the list of dictionaries format
    metadata = df_chunks.to_dict(orient='records')

    # Add a unique ID to each chunk for better tracking, if not present
    for i, item in enumerate(metadata):
        item['chunk_id'] = item.get('chunk_id', f'chunk_{i}')

    # Call the main function to create and save the FAISS index
    create_and_save_faiss_index(
        metadata=metadata,
        model_name=embedding_model_name,
        index_path=faiss_index_path,
        metadata_path=metadata_path
    )

def prepare_training_data(
        dataset_path,
        embedding_model_name,
        use_subset,
        subset_size,
        train_data_path,
        val_data_path,
        test_data_path,
        kb_chunks_path
    ):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    if use_subset:
        # Using a sample for faster execution
        df_original = pd.read_csv(dataset_path).dropna().sample(n=subset_size, random_state=42)
    else:
        df_original = pd.read_csv(dataset_path)

    # Create the datasets with train/val/test split
    unique_chunks, train_data, val_data, test_data = create_squad_dataset_pipeline(
        df_original, tokenizer,
        train_ratio=0.7,    # 70% for training
        val_ratio=0.15,     # 15% for validation
        test_ratio=0.15     # 15% for test
    )

    print("\n--- Example of Generated Training Data ---")
    if train_data:
        print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
    
    # Save all three datasets
    print(f"\n--- Saving Datasets ---")
    
    # Save the unified knowledge base chunks
    df_chunks = pd.DataFrame(unique_chunks, columns=['answer_chunk'])
    df_chunks.to_csv(kb_chunks_path, index=False)
    print(f"Knowledge base chunks saved to '{kb_chunks_path}'")

    # Save training data
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"Training dataset saved to '{train_data_path}'")
    
    # Save validation data
    with open(val_data_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    print(f"Validation dataset saved to '{val_data_path}'")
    
    # Save test data
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"Test dataset saved to '{test_data_path}'")
    
    print(f"\n--- Summary ---")
    print(f"Knowledge base chunks: {len(unique_chunks)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Total examples: {len(train_data) + len(val_data) + len(test_data)}")

def train_model(
        train_data_path,
        val_data_path,
        test_data_path,
        model_checkpoint,
        model_path,
        training_summary_path
    ):
    # --- Step 1: Load Everything ---
    print("Loading components...")
    
    # Load the three separate datasets
    print("Loading train/validation/test datasets...")
    train_raw = load_dataset("json", data_files=train_data_path)["train"]
    val_raw = load_dataset("json", data_files=val_data_path)["train"]
    test_raw = load_dataset("json", data_files=test_data_path)["train"]
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_raw)} examples")
    print(f"  Validation: {len(val_raw)} examples")
    print(f"  Test: {len(test_raw)} examples")
    
    # Define the model to use as the base for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    # --- Step 2: Preprocess the Data ---
    print("Preprocessing data for model format...")
    
    # Process each dataset separately
    train_dataset = train_raw.map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names
    )
    
    validation_dataset = val_raw.map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names
    )
    
    test_dataset = test_raw.map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=test_raw.column_names
    )
    
    print(f"Preprocessed dataset sizes:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(validation_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    # --- Step 3: Set Up Training Arguments ---
    print("Setting up training arguments...")
    
    # Check if CUDA is available and configure accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Optimize batch size for GPU (RTX 3050 Ti has ~4GB VRAM)
    batch_size = 8 if torch.cuda.is_available() else 4
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False, # Do not push to Hugging Face Hub
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU for faster training
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        logging_steps=10,  # Log more frequently
        save_steps=50,  # Save checkpoints more frequently
        warmup_steps=100,  # Warmup for better training stability
    )

    # --- Step 4: Instantiate and Start the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,  # Use validation dataset for evaluation during training
        tokenizer=tokenizer,
    )

    print("\n--- STARTING TRAINING ---")
    trainer.train()
    print("--- TRAINING COMPLETED ---")

    # --- Step 5: Final Evaluation on Test Set ---
    print("\n--- EVALUATING ON TEST SET ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # --- Step 6: Save the Final Model ---
    
    print(f"\nSaving the final model to '{model_path}'...")
    trainer.save_model(model_path)
    print("Model saved successfully!")
    
    # --- Step 7: Save Training Summary ---
    summary = {
        "model_checkpoint": model_checkpoint,
        "dataset_splits": {
            "train": len(train_dataset),
            "validation": len(validation_dataset), 
            "test": len(test_dataset)
        },
        "training_args": {
            "batch_size": batch_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "fp16": training_args.fp16,
            "device": device
        },
        "test_results": test_results
    }
    
    with open(training_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to '{training_summary_path}'")


if __name__ == "__main__":
    from config import *

    print("="*100, "\nDOWNLOADING DATA\n" + "="*100)
    download_data(
        data_source=DATA_SOURCE,
        dataset_path=DATASET_PATH
    )

    print("="*100, "\nPREPARING VECTOR DATABASE\n" + "="*100)
    prepare_vector_database(
        dataset_path=DATASET_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        faiss_index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        use_subset=USE_SUBSET,
        subset_size=SUBSET_SIZE
    )

    print("="*100, "\nPREPARING TRAINING DATA\n" + "="*100)
    prepare_training_data(
        dataset_path=DATASET_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        use_subset=USE_SUBSET,
        subset_size=SUBSET_SIZE,
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        kb_chunks_path=KB_CHUNKS_PATH
    )

    print("="*100, "\nTRAINING MODEL\n" + "="*100)
    train_model(
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        model_checkpoint=MODEL_CHECKPOINT,
        model_path=MODEL_PATH,
        training_summary_path=TRAINING_SUMMARY_PATH
    )