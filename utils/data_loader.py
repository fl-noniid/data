"""
Updated Advanced Data Loader with correct HF paths and Global Corpus logic
"""
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any, Optional, Tuple
import random
import time


class AdvancedDataLoader:
    """Loader for various QA and conversation datasets with global corpus support"""
    
    DATASET_CONFIGS = {
        "squad": {"path": "rajpurkar/squad", "split": "validation", "type": "single-step"},
        "nq": {"path": "florin-hf/nq_open_gold", "split": "validation", "type": "single-step"},
        "triviaqa": {"path": "mandarjoshi/trivia_qa", "name": "rc.nocontext", "split": "validation", "type": "single-step"},
        "musique": {"path": "bdsaglam/musique", "split": "validation", "type": "multi-step"},
        "hotpotqa": {"path": "hotpotqa/hotpot_qa", "name": "distractor", "split": "validation", "type": "multi-step"},
        "2wikimultihopqa": {"path": "framolfese/2WikiMultihopQA", "split": "validation", "type": "multi-step"},
        "sharegpt": {"path": "Aeala/ShareGPT_Vicuna_unfiltered", "split": "train", "type": "no-retrieval"}
    }
    
    def __init__(self, dataset_names: List[str], sample_size: Optional[int] = None):
        """
        Initialize data loader with multiple datasets
        
        Args:
            dataset_names: List of dataset names to load
            sample_size: Total number of samples to randomly pick from the mixed pool
        """
        self.all_examples = []
        self.global_corpus = []
        
        for name in dataset_names:
            if name not in self.DATASET_CONFIGS:
                print(f"Warning: Unsupported dataset {name}, skipping.")
                continue
                
            config = self.DATASET_CONFIGS[name]
            print(f"Loading {name} dataset from {config['path']}...")
            
            try:
                load_args = {"path": config["path"]}
                if "name" in config:
                    load_args["name"] = config["name"]
                load_args["split"] = config["split"]
                
                ds = load_dataset(**load_args)
                
                # 1. Extract Global Corpus (All samples regardless of sample_size)
                self._extract_corpus(ds, name)
                
                # 2. Format and add to mixed pool
                formatted_ds = self._format_dataset(ds, name, config["type"])
                self.all_examples.extend(formatted_ds)
                
            except Exception as e:
                print(f"Error loading {name}: {e}")

        # 3. Randomly sample from the mixed pool
        if sample_size and sample_size < len(self.all_examples):
            self.all_examples = random.sample(self.all_examples, sample_size)
            
        print(f"Total mixed examples: {len(self.all_examples)}")
        print(f"Total global corpus size: {len(self.global_corpus)}")

    def _extract_corpus(self, ds, name):
        """Extract all possible context/text for the global corpus"""
        if name == "squad":
            self.global_corpus.extend(list(set(ds["context"])))
        elif name == "hotpotqa":
            for ctx in ds["context"]:
                for title, sentences in zip(ctx["title"], ctx["sentences"]):
                    self.global_corpus.append(f"{title}: {' '.join(sentences)}")
        elif name == "musique":
            for paragraphs in ds["paragraphs"]:
                for p in paragraphs:
                    self.global_corpus.append(f"{p['title']}: {p['paragraph_text']}")
        elif name == "2wikimultihopqa":
            for context in ds["context"]:
                for title, sentences in context:
                    self.global_corpus.append(f"{title}: {' '.join(sentences)}")
        elif "context" in ds.column_names:
            self.global_corpus.extend([str(c) for c in ds["context"] if c])
        
        # Deduplicate
        self.global_corpus = list(set(self.global_corpus))

    def _format_dataset(self, ds, name, ds_type) -> List[Dict[str, Any]]:
        formatted = []
        for i in range(len(ds)):
            ex = ds[i]
            item = {"dataset": name, "type": ds_type, "id": str(ex.get("id", i))}
            
            if name == "squad":
                item["question"] = ex["question"]
                item["answer"] = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
            elif name == "nq":
                item["question"] = ex["question"]
                item["answer"] = ex["answer"][0] if ex["answer"] else ""
            elif name == "sharegpt":
                convs = ex["conversations"]
                item["question"] = next((m["value"] for m in convs if m["from"] == "human"), "")
                item["answer"] = next((m["value"] for m in convs if m["from"] == "gpt"), "")
            else:
                item["question"] = ex.get("question", "")
                item["answer"] = ex.get("answer", "")
                
            if item["question"]:
                formatted.append(item)
        return formatted

    def get_examples(self) -> List[Dict[str, Any]]:
        return self.all_examples

    def get_global_corpus(self) -> List[str]:
        return self.global_corpus
