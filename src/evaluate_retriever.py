import faiss
import numpy as np
import os
import pickle
import re
import string
import time

from collections import Counter
from config import *
from datasets import load_dataset
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Helper from evaluate.py to normalize text
def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def load_retriever_components(
        faiss_index_path,
        metadata_path,
        embedding_model_name,
        dataset_path
        ):
    """Load all required components for retriever evaluation."""
    print("Loading components for retriever evaluation...")
    
    # Load FAISS index and metadata
    index = faiss.read_index(faiss_index_path)
    embedding_model = SentenceTransformer(embedding_model_name)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    return {
        'embedding_model': embedding_model,
        'index': index,
        'metadata': metadata,
        'dataset': dataset
    }


def evaluate_retriever_performance(
        dataset,
        embedding_model,
        index,
        metadata,
        k=3,
        max_examples=None
    ):
    """
    Evaluates the retriever's performance by checking if the correct answer 
    is found in the top-k retrieved documents.
    """
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    hits = 0
    misses = []
    
    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing example {i+1}/{len(dataset)}...")

        question = example['question']
        gold_answers = example['answers']['text']
        
        # 1. Embed the question
        question_embedding = embedding_model.encode([question], convert_to_numpy=True)
        
        # 2. Search the index
        _, top_k_indices = index.search(question_embedding, k)
        
        # 3. Get retrieved full answers
        retrieved_answers = [metadata[j]['answer'] for j in top_k_indices[0]]
        retrieved_chunks = [metadata[j]['answer_chunk'] for j in top_k_indices[0]]

        # 4. Check for a hit
        is_hit = False
        for retrieved_answer in retrieved_answers:
            normalized_retrieved = normalize_answer(retrieved_answer)
            for gold_answer in gold_answers:
                normalized_gold = normalize_answer(gold_answer)
                if normalized_gold in normalized_retrieved:
                    is_hit = True
                    break
            if is_hit:
                break
        
        if is_hit:
            hits += 1
        else:
            misses.append({
                'question': question,
                'gold_answers': gold_answers,
                'retrieved_chunks': retrieved_chunks
            })

    recall = (hits / len(dataset)) * 100 if len(dataset) > 0 else 0
    
    return {
        'recall_at_k': recall,
        'k': k,
        'num_examples': len(dataset),
        'hits': hits,
        'misses': len(misses),
        'missed_examples': misses
    }

def generate_retriever_report(run_name, results, embedding_model_name, k):
    """Generate a markdown report for retriever evaluation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = (
        f"# Retriever Evaluation Report - {run_name}\n"
        f"**Generated:** {timestamp}\n"
        f"**Embedding Model:** {embedding_model_name}\n"
        f"**Evaluation Size:** {results['num_examples']} examples\n"
        f"**K (top retrieved):** {k}\n"
    )

    report += (
        "## Summary\n"
        f"- **Recall@{k}:** {results['recall_at_k']:.2f}%\n"
        f"- **Total Questions:** {results['num_examples']}\n"
        f"- **Hits (correct document retrieved):** {results['hits']}\n"
        f"- **Misses:** {results['misses']}\n"
    )

    # Examples of misses
    if results['missed_examples']:
        report += "\n## Examples of Retrieval Misses\n"
        report += "Below are questions where the correct answer was not found in the retrieved documents.\n\n"
        
        for i, ex in enumerate(results['missed_examples'][:5]): # Show up to 5 examples
            report += f"### Miss Example {i+1}\n"
            report += f"- **Question:** {ex['question']}\n"
            report += f"- **Gold Answers:** {ex['gold_answers']}\n"
            report += "**Retrieved Chunks:**\n"
            for chunk_idx, chunk in enumerate(ex['retrieved_chunks']):
                report += f"  - Chunk {chunk_idx+1}: `{chunk}`\n"
            report += "\n"
            
    return report

def save_retriever_report(reports_dir, report_content):
    """Save the markdown report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"retriever_evaluation_{timestamp}.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    return report_path

def evaluate_retriever(
        run_name,
        embedding_model_name,
        faiss_index_path,
        metadata_path,
        test_data_path,
        report_dir,
        max_examples,
        k
    ):
    """Evaluate the retriever performance."""
    print("--- Starting Retriever Evaluation ---", "\n" + "="*100)
    
    components = load_retriever_components(
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        embedding_model_name=embedding_model_name,
        dataset_path=test_data_path # Evaluating on test set
    )
    
    print(f"\nEvaluating retriever with k={k} on {max_examples} examples...")
    
    results = evaluate_retriever_performance(
        dataset=components['dataset'],
        embedding_model=components['embedding_model'],
        index=components['index'],
        metadata=components['metadata'],
        k=k,
        max_examples=max_examples
    )
    
    print("\n--- Evaluation Complete ---")
    print(f"Recall@{k}: {results['recall_at_k']:.2f}%")
    
    report_content = generate_retriever_report(
        run_name=run_name,
        results=results,
        embedding_model_name=embedding_model_name,
        k=k
    )
    
    report_path = save_retriever_report(report_dir, report_content)
    
    print(f"\nRetriever evaluation report saved to: {report_path}") 


    