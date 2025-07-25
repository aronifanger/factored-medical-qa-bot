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
