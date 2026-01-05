"""
Advanced Adaptive RAG Main Execution Script
"""
import argparse
import json
import os
from tqdm import tqdm
from tabulate import tabulate

from utils.vllm_client import vLLMClient
from utils.data_loader import AdvancedDataLoader
from utils.rag_components import DenseRetriever
from models.adaptive_rag import AdvancedAdaptiveRAG


def main():
    parser = argparse.ArgumentParser(description="Advanced Adaptive RAG Application")
    
    # Environment settings
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["squad", "nq", "triviaqa", "musique", "hotpotqa", "2wikimultihopqa", "sharegpt"],
                        help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max-steps", type=int, default=3, help="Max steps for multi-step RAG")
    
    # Output settings
    parser.add_argument("--output", type=str, default="advanced_results.json")
    
    args = parser.parse_args()

    # 1. Initialize Components
    print(f"\n--- Initializing Components for {args.dataset} ---")
    vllm_client = vLLMClient(base_url=args.vllm_url, model_name=args.vllm_model)
    retriever = DenseRetriever(embedding_model=args.embedding_model)
    
    # 2. Load Data and Build Global Corpus
    data_loader = AdvancedDataLoader(args.dataset, sample_size=args.num_samples)
    
    if data_loader.dataset_type != "no-retrieval":
        corpus = data_loader.get_global_corpus()
        retriever.build_global_index(corpus)
    
    # 3. Initialize RAG System
    rag_system = AdvancedAdaptiveRAG(vllm_client, retriever, max_steps=args.max_steps)
    
    # 4. Run Evaluation
    results = []
    print(f"\n--- Running Evaluation ({args.num_samples} samples) ---")
    
    for i in tqdm(range(len(data_loader))):
        example = data_loader.get_example(i)
        
        # Run pipeline
        output = rag_system.run(example["question"], example["type"])
        
        # Add ground truth and metadata
        output.update({
            "id": example["id"],
            "question": example["question"],
            "ground_truth": example.get("answer", ""),
            "dataset": args.dataset
        })
        
        results.append(output)

    # 5. Print Results and Metrics
    print_summary(results)
    
    # 6. Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {args.output}")


def print_summary(results):
    """Print a formatted summary table of the results"""
    if not results:
        return

    table_data = []
    headers = ["ID", "Strategy", "E2E Latency", "LLM Calls", "LLM Latency", "Ret Latency", "Steps"]
    
    total_e2e = 0
    total_llm_calls = 0
    
    for r in results:
        table_data.append([
            r["id"][:8],
            r["strategy"],
            f"{r['end_to_end_latency']:.3f}s",
            r["llm_calls"],
            f"{r['llm_latency']:.3f}s",
            f"{r['retrieval_latency']:.3f}s",
            r["steps"]
        ])
        total_e2e += r["end_to_end_latency"]
        total_llm_calls += r["llm_calls"]

    print("\n" + "="*80)
    print("DETAILED PERFORMANCE METRICS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    avg_e2e = total_e2e / len(results)
    avg_calls = total_llm_calls / len(results)
    
    print("\n" + "="*30)
    print("AGGREGATED STATISTICS")
    print("="*30)
    print(f"Average End-to-End Latency: {avg_e2e:.3f}s")
    print(f"Average LLM Calls per Query: {avg_calls:.2f}")
    print(f"Total Samples: {len(results)}")
    print("="*30)


if __name__ == "__main__":
    main()
