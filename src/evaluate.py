import faiss
import numpy as np
import os
import pickle
import re
import string
import torch

from collections import Counter
from config import *
from datasets import load_dataset
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_dataset(
        dataset,
        qa_pipeline,
        embedding_model,
        index,
        metadata,
        use_retrieval=False,
        max_examples=None
    ):
    """Evaluate the model on a dataset."""
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    exact_matches = []
    f1_scores = []
    examples = []
    
    for example in dataset:
        question = example['question']
        gold_answers = example['answers']['text']
        
        if use_retrieval:
            question_embedding = embedding_model.encode([question], convert_to_numpy=True)
            distances, indices = index.search(question_embedding, k=3)
            context = " ".join([metadata[j]['answer_chunk'] for j in indices[0]])
        else:
            context = example['context']
        
        try:
            result = qa_pipeline(question=question, context=context)
            predicted_answer = result['answer']
            confidence = result['score']
        except Exception:
            predicted_answer = ""
            confidence = 0.0
        
        # Compute metrics against all gold answers (take max)
        max_exact = 0
        max_f1 = 0
        for gold_answer in gold_answers:
            exact = compute_exact(gold_answer, predicted_answer)
            f1 = compute_f1(gold_answer, predicted_answer)
            max_exact = max(max_exact, exact)
            max_f1 = max(max_f1, f1)
        
        exact_matches.append(max_exact)
        f1_scores.append(max_f1)
        
        # Store example for reporting
        examples.append({
            'question': question,
            'predicted_answer': predicted_answer,
            'gold_answers': gold_answers,
            'confidence': confidence,
            'exact_match': max_exact,
            'f1_score': max_f1,
            'context': context
        })
    
    return {
        'exact_match': np.mean(exact_matches) * 100,
        'f1_score': np.mean(f1_scores) * 100,
        'num_examples': len(exact_matches),
        'examples': examples
    }


def load_components(
        model_path,
        faiss_index_path,
        metadata_path,
        train_data_path,
        val_data_path,
        test_data_path,
        embedding_model_name
        ):
    """Load all required components for evaluation."""
    print("Loading components...")
    
    # Load FAISS index and metadata
    index = faiss.read_index(faiss_index_path)
    embedding_model = SentenceTransformer(embedding_model_name)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Load QA model
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=train_data_path)["train"]
    val_dataset = load_dataset("json", data_files=val_data_path)["train"]  
    test_dataset = load_dataset("json", data_files=test_data_path)["train"]
    
    return {
        'qa_pipeline': qa_pipeline,
        'embedding_model': embedding_model,
        'index': index,
        'metadata': metadata,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def run_evaluation(
        model_path,
        faiss_index_path,
        metadata_path,
        train_data_path,
        val_data_path,
        test_data_path,
        embedding_model_name,
        max_examples=100
    ):
    """Run comprehensive evaluation and return results."""
    components = load_components(
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        embedding_model_name=embedding_model_name
    )
    
    results = {}
    
    # Evaluate all datasets
    for dataset_name, dataset in components['datasets'].items():
        print(f"Evaluating {dataset_name} dataset...")
        
        # Direct QA
        results[f'{dataset_name}_direct'] = evaluate_dataset(
            dataset, 
            components['qa_pipeline'],
            components['embedding_model'],
            components['index'],
            components['metadata'],
            use_retrieval=False, 
            max_examples=max_examples
        )
        
        # Retrieval + QA
        results[f'{dataset_name}_retrieval'] = evaluate_dataset(
            dataset,
            components['qa_pipeline'], 
            components['embedding_model'],
            components['index'],
            components['metadata'],
            use_retrieval=True, 
            max_examples=max_examples
        )
    
    return results


def get_example_predictions(results, num_examples=2):
    """Get correct and incorrect examples for reporting."""
    examples_section = ""
    
    # Get examples from test set (most important)
    test_direct = results.get('test_direct', {}).get('examples', [])
    test_retrieval = results.get('test_retrieval', {}).get('examples', [])
    
    if test_direct:
        examples_section += "\n## Example Predictions\n\n"
        
        # Correct examples from Direct QA
        correct_examples = [ex for ex in test_direct if ex['exact_match'] == 1]
        incorrect_examples = [ex for ex in test_direct if ex['exact_match'] == 0]
        
        if correct_examples:
            examples_section += "### Correct Predictions (Direct QA)\n\n"
            for i, ex in enumerate(correct_examples[:num_examples]):
                examples_section += f"**Example {i+1}:**\n"
                examples_section += f"- Question: {ex['question']}\n"
                examples_section += f"- Gold Answer: {ex['gold_answers'][0]}\n"
                examples_section += f"- Predicted: {ex['predicted_answer']}\n"
                examples_section += f"- Confidence: {ex['confidence']:.3f}\n"
                examples_section += f"- Context: {ex['context']}\n\n"
        
        if incorrect_examples:
            examples_section += "### Incorrect Predictions (Direct QA)\n\n"
            for i, ex in enumerate(incorrect_examples[:num_examples]):
                examples_section += f"**Example {i+1}:**\n"
                examples_section += f"- Question: {ex['question']}\n"
                examples_section += f"- Gold Answer: {ex['gold_answers'][0]}\n"
                examples_section += f"- Predicted: {ex['predicted_answer']}\n"
                examples_section += f"- Confidence: {ex['confidence']:.3f}\n"
                examples_section += f"- F1 Score: {ex['f1_score']:.2f}\n"
                examples_section += f"- Context: {ex['context']}\n\n"
        
        # Retrieval examples
        if test_retrieval:
            retrieval_incorrect = [ex for ex in test_retrieval if ex['exact_match'] == 0]
            if retrieval_incorrect:
                examples_section += "### Retrieval+QA Failures\n\n"
                for i, ex in enumerate(retrieval_incorrect[:1]):  # Just one example
                    examples_section += f"**Example {i+1}:**\n"
                    examples_section += f"- Question: {ex['question']}\n"
                    examples_section += f"- Gold Answer: {ex['gold_answers'][0]}\n"
                    examples_section += f"- Predicted: {ex['predicted_answer']}\n"
                    examples_section += f"- Retrieved Context: {ex['context']}\n\n"
    
    return examples_section


def generate_markdown_report(
        run_name,
        model_path,
        embedding_model_name,
        results,
        max_examples=100
    ):
    """Generate a markdown report from evaluation results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = (
        f"# Model Evaluation Report - {run_name}\n"
        f"**Generated:** {timestamp}\n"
        f"**Model Path:** {model_path}\n"
        f"**Embedding Model:** {embedding_model_name}\n"
        f"**Evaluation Size:** {max_examples} examples per dataset\n"
        f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}\n"
    )
    
    report += (
        "## Summary\n"
        "| Dataset | Method | Exact Match | F1 Score | Examples |\n"
        "|---------|--------|-------------|----------|----------|\n"
    )
    
    for key, result in results.items():
        dataset_name, method = key.split('_')
        dataset_name = dataset_name.capitalize()
        method = "Direct QA" if method == "direct" else "Retrieval+QA"
        
        report += f"| {dataset_name} | {method} | {result['exact_match']:.2f}% | {result['f1_score']:.2f}% | {result['num_examples']} |\n"
    
    # Analysis
    test_direct_f1 = results.get('test_direct', {}).get('f1_score', 0)
    test_retrieval_f1 = results.get('test_retrieval', {}).get('f1_score', 0)
    train_direct_f1 = results.get('train_direct', {}).get('f1_score', 0)
    
    report += (
        "## Analysis\n"
        f"**Test Performance:**\n"
        f"- Direct QA F1: {test_direct_f1:.2f}%\n"
        f"- Retrieval+QA F1: {test_retrieval_f1:.2f}%\n"
        f"**Overfitting Check:**\n"
        f"- Training F1: {train_direct_f1:.2f}%\n"
        f"- Test F1: {test_direct_f1:.2f}%\n"
        f"- Gap: {train_direct_f1 - test_direct_f1:.2f}%\n"
    )
    
    # Add examples section
    examples_section = get_example_predictions(results)
    report += examples_section
    
    return report


def save_report(reports_dir, report_content):
    """Save the markdown report to the reports directory."""
    # Create reports directory structure
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"evaluation_{timestamp}.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


def evaluate_model(
        run_name,
        model_path,
        run_description,
        embedding_model_name,
        faiss_index_path,
        metadata_path,
        train_data_path,
        val_data_path,
        test_data_path,
        report_dir,
        max_examples=100
    ):
    """Main evaluation function."""
    print(f"Starting evaluation for {run_name}")
    print(f"Run description: {run_description}")
    print(f"Max examples per dataset: {max_examples}")
    
    # Run evaluation
    results = run_evaluation(
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        embedding_model_name=embedding_model_name,
        max_examples=max_examples
    )
    
    # Generate report
    report_content = generate_markdown_report(
        run_name=run_name,
        model_path=model_path,
        embedding_model_name=embedding_model_name,
        results=results,
        max_examples=max_examples
    )
    
    # Save report
    report_path = save_report(report_dir, report_content)
    
    print(f"Evaluation complete. Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    from config import *
    evaluate_model(
        run_name=RUN,
        model_path=MODEL_PATH,
        run_description=RUN_DESCRIPTION,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        faiss_index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        report_dir=REPORTS_DIR,
        max_examples=10
    )