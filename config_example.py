"""
Configuration example for Adaptive RAG Application

Copy this file to config.py and modify as needed.
"""

# vLLM Server Configuration
VLLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "temperature": 0.7,
    "max_tokens": 512
}

# Alternative models you can use with vLLM:
# - "meta-llama/Llama-2-13b-chat-hf"
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "tiiuae/falcon-7b-instruct"

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu"  # or "cuda" if GPU is available
}

# Alternative embedding models:
# - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)

# Retriever Configuration
RETRIEVER_CONFIG = {
    "top_k": 3,  # Number of documents to retrieve
    "score_threshold": None  # Optional: minimum similarity score
}

# Dataset Configuration
DATASET_CONFIG = {
    "name": "hotpot_qa",
    "config": "distractor",
    "split": "validation",
    "sample_size": 100  # None for full dataset
}

# RAG Strategy Configuration
RAG_CONFIG = {
    "routing_strategy": "random",  # Currently only "random" is supported
    "multi_step_iterations": 2  # Number of steps for multi-step RAG
}

# Evaluation Configuration
EVAL_CONFIG = {
    "output_dir": "./results",
    "save_intermediate": True,
    "verbose": True
}
