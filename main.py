"""
Advanced Adaptive RAG Main Execution Script - Updated for Mixed Datasets
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
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["squad", "nq", "triviaqa", "musique", "hotpotqa", "2wikimultihopqa", "sharegpt"],
                        help="List of datasets to mix and sample from")
    parser.add_argument("--num-samples", type=int, default=20, help="Total number of samples to evaluate from the mixed pool")
    parser.add_argument("--max-steps", type=int, default=3, help="Max steps for multi-step RAG")
    
    # Output settings
    parser.add_argument("--output", type=str, default="final_results.json")
    
    args = parser.parse_args()

    # 1. Initialize Components
    print(f"\n--- Initializing Components ---")
    vllm_client = vLLMClient(base_url=args.vllm_url, model_name=args.vllm_model)
    retriever = DenseRetriever(embedding_model=args.embedding_model)
    
    # 2. Load Data and Build Global Corpus (All samples from all datasets)
    data_loader = AdvancedDataLoader(args.datasets, sample_size=args.num_samples)
    
    corpus = data_loader.get_global_corpus()
    if corpus:
        retriever.build_global_index(corpus)
    
    # 3. Initialize RAG System
    rag_system = AdvancedAdaptiveRAG(vllm_client, retriever, max_steps=args.max_steps)
    
    # 4. Run Evaluation on Mixed Samples
    results = []
    examples = data_loader.get_examples()
    print(f"\n--- Running Evaluation on {len(examples)} Mixed Samples ---")
    
    for ex in tqdm(examples):
        # Run pipeline based on the original dataset type
        output = rag_system.run(ex["question"], ex["type"])
        
        # Add metadata
        output.update({
            "id": ex["id"],
            "question": ex["question"],
            "ground_truth": ex.get("answer", ""),
            "dataset": ex["dataset"]
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
    headers = ["Dataset", "Strategy", "E2E Lat", "LLM Calls", "LLM Lat", "Ret Lat", "Steps"]
    
    for r in results:
        table_data.append([
            r["dataset"],
            r["strategy"],
            f"{r['end_to_end_latency']:.2f}s",
            r["llm_calls"],
            f"{r['llm_latency']:.2f}s",
            f"{r['retrieval_latency']:.2f}s",
            r["steps"]
        ])

    print("\n" + "="*90)
    print("DETAILED PERFORMANCE METRICS (MIXED DATASETS)")
    print("="*90)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Aggregated stats by dataset
    print("\n" + "="*40)
    print("AGGREGATED STATISTICS BY DATASET")
    print("="*40)
    datasets = sorted(list(set(r["dataset"] for r in results)))
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        avg_e2e = sum(r["end_to_end_latency"] for r in ds_results) / len(ds_results)
        print(f"{ds:15}: Avg E2E {avg_e2e:.2f}s | Samples: {len(ds_results)}")
    print("="*40)


if __name__ == "__main__":
    main()
