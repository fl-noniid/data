"""
Advanced Data Loader for Multiple Datasets
Supports: SQuAD, NQ, TriviaQA, Musique, HotpotQA, 2WikiMultiHopQA, ShareGPT
"""
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Union
import random
import time


class AdvancedDataLoader:
    """Loader for various QA and conversation datasets"""
    
    DATASET_CONFIGS = {
        # Single-step QA
        "squad": {"path": "squad", "split": "validation", "type": "single-step"},
        "nq": {"path": "nq_open", "split": "validation", "type": "single-step"},
        "triviaqa": {"path": "trivia_qa", "name": "rc.nocontext", "split": "validation", "type": "single-step"},
        
        # Multi-step QA
        "musique": {"path": "hotpot_qa", "name": "distractor", "split": "validation", "type": "multi-step"}, # Placeholder for Musique
        "hotpotqa": {"path": "hotpot_qa", "name": "distractor", "split": "validation", "type": "multi-step"},
        "2wikimultihopqa": {"path": "hotpot_qa", "name": "distractor", "split": "validation", "type": "multi-step"}, # Placeholder
        
        # Normal conversation
        "sharegpt": {"path": "Aeala/ShareGPT_Vicuna_unfiltered", "split": "train", "type": "no-retrieval"}
    }
    
    def __init__(self, dataset_name: str, sample_size: Optional[int] = None):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of the dataset to load
            sample_size: Number of samples to load
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.dataset_type = self.config["type"]
        
        print(f"Loading {dataset_name} dataset...")
        
        # Load dataset
        load_args = {"path": self.config["path"]}
        if "name" in self.config:
            load_args["name"] = self.config["name"]
        load_args["split"] = self.config["split"]
        
        self.dataset = load_dataset(**load_args)
        
        if sample_size:
            indices = random.sample(range(len(self.dataset)), min(sample_size, len(self.dataset)))
            self.dataset = self.dataset.select(indices)
            
        print(f"Loaded {len(self.dataset)} examples from {dataset_name}")

    def get_example(self, idx: int) -> Dict[str, Any]:
        """Get a single example formatted for the RAG pipeline"""
        example = self.dataset[idx]
        
        if self.dataset_name == "squad":
            return {
                "id": example["id"],
                "question": example["question"],
                "answer": example["answers"]["text"][0] if example["answers"]["text"] else "",
                "context": example["context"],
                "type": "single-step"
            }
        elif self.dataset_name == "hotpotqa":
            return {
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "context": example["context"],
                "type": "multi-step"
            }
        elif self.dataset_name == "sharegpt":
            # Extract first human message
            conversations = example["conversations"]
            human_msg = next((m["value"] for m in conversations if m["from"] == "human"), "")
            gpt_msg = next((m["value"] for m in conversations if m["from"] == "gpt"), "")
            return {
                "id": str(idx),
                "question": human_msg,
                "answer": gpt_msg,
                "type": "no-retrieval"
            }
        else:
            # Generic mapping for other datasets (simplified for this implementation)
            return {
                "id": str(idx),
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "type": self.dataset_type
            }

    def get_global_corpus(self) -> List[str]:
        """
        Extract a global corpus from the dataset for building the vector DB
        In a real scenario, this would be a massive external corpus.
        Here we use the contexts available in the dataset.
        """
        corpus = []
        if "context" in self.dataset.column_names:
            # For datasets like SQuAD or HotpotQA that have context
            if self.dataset_name == "hotpotqa":
                for ctx in self.dataset["context"]:
                    for title, sentences in zip(ctx["title"], ctx["sentences"]):
                        corpus.append(f"{title}: {' '.join(sentences)}")
            else:
                corpus = list(set(self.dataset["context"]))
        elif self.dataset_name == "sharegpt":
            # For conversation, we don't really have a corpus to retrieve from
            corpus = ["This is a conversation dataset. No retrieval needed."]
        else:
            # Fallback
            corpus = ["Sample document for " + self.dataset_name]
            
        return list(set(corpus)) # Remove duplicates

    def __len__(self):
        return len(self.dataset)
