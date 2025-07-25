import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

def prepare_train_features(examples, tokenizer):
    """
    Tokenizes the texts and maps answers (in character positions)
    to the token positions required by the model.
    """
    # Remove leading/trailing whitespace from questions
    examples["question"] = [q.strip() for q in examples["question"]]
    
    # Tokenize (question, context) pairs
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # Only truncate the context if too long
        max_length=512,
        stride=128,  # Sliding window with 128 token overlap
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # Needed for mapping tokens to character positions
        padding="max_length",
    )

    # The tokenizer can create multiple features from a long example.
    # We need to map which feature came from which original example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Add the labels (start_positions and end_positions)
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Identify which original example this feature belongs to
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers are present (e.g., negative examples), set CLS as the answer
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Get the start and end of the answer in CHARACTER positions
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the start and end of the CONTEXT in TOKEN positions
            token_start_index = 0
            while tokenized_examples.sequence_ids(i)[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while tokenized_examples.sequence_ids(i)[token_end_index] != 1:
                token_end_index -= 1

            # Check if the answer is completely outside the context span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise, find the start and end token indices of the answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# --- Main Execution Block ---
if __name__ == "__main__":
    from config import *

    # --- Step 1: Load Everything ---
    print("Loading components...")
    
    # Load the three separate datasets
    print("Loading train/validation/test datasets...")
    train_raw = load_dataset("json", data_files=TRAIN_DATA_PATH)["train"]
    val_raw = load_dataset("json", data_files=VAL_DATA_PATH)["train"]
    test_raw = load_dataset("json", data_files=TEST_DATA_PATH)["train"]
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_raw)} examples")
    print(f"  Validation: {len(val_raw)} examples")
    print(f"  Test: {len(test_raw)} examples")
    
    # Define the model to use as the base for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

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
    
    print(f"\nSaving the final model to '{MODEL_PATH}'...")
    trainer.save_model(MODEL_PATH)
    print("Model saved successfully!")
    
    # --- Step 7: Save Training Summary ---
    summary = {
        "model_checkpoint": MODEL_CHECKPOINT,
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
    
    import json
    summary_path = TRAINING_SUMMARY_PATH
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to '{summary_path}'")