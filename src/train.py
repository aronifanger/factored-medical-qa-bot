from transformers import TrainerCallback
from datetime import datetime
import os


class CompactLoggingCallback(TrainerCallback):
    """Custom callback for compact training logs."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logs = []
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize logging at training start."""
        self.start_time = datetime.now()
        # Create header
        header = "# Training Log\n\n"
        header += f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"**Model:** {args.output_dir}\n\n"
        header += "## Training Progress\n\n"
        header += "| Epoch | Step | Train Loss | Eval Loss | Eval Runtime | Learning Rate |\n"
        header += "|-------|------|------------|-----------|--------------|---------------|\n"
        
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        with open(self.log_file_path, 'w') as f:
            f.write(header)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics at specified intervals."""
        if logs is None:
            return
            
        # Only log every few steps to reduce verbosity
        if state.global_step % 50 == 0 or 'eval_loss' in logs:
            log_entry = {
                'epoch': round(state.epoch, 2),
                'step': state.global_step,
                'train_loss': logs.get('train_loss', ''),
                'eval_loss': logs.get('eval_loss', ''),
                'eval_runtime': logs.get('eval_runtime', ''),
                'learning_rate': logs.get('learning_rate', '')
            }
            
            # Format the log entry
            if log_entry['train_loss'] or log_entry['eval_loss']:
                line = f"| {log_entry['epoch']:.1f} | {log_entry['step']} | "
                line += f"{log_entry['train_loss']:.4f} | " if log_entry['train_loss'] else "- | "
                line += f"{log_entry['eval_loss']:.4f} | " if log_entry['eval_loss'] else "- | "
                line += f"{log_entry['eval_runtime']:.1f}s | " if log_entry['eval_runtime'] else "- | "
                line += f"{log_entry['learning_rate']:.2e} |\n" if log_entry['learning_rate'] else "- |\n"
                
                # Append to file
                with open(self.log_file_path, 'a') as f:
                    f.write(line)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize logging at training end."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        footer = f"\n## Training Summary\n\n"
        footer += f"**Completed:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"**Duration:** {str(duration).split('.')[0]}\n"
        footer += f"**Total Steps:** {state.global_step}\n"
        footer += f"**Final Epoch:** {state.epoch:.1f}\n"
        
        with open(self.log_file_path, 'a') as f:
            f.write(footer)




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
