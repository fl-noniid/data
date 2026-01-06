"""
Advanced Adaptive RAG with IR-CoT and Detailed Metrics
"""
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from langchain.schema import Document

from utils.vllm_client import vLLMClient
from utils.rag_components import DenseRetriever, format_context


class RAGStrategy(Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


class AdvancedAdaptiveRAG:
    """Advanced Adaptive RAG system with IR-CoT and performance tracking"""
    
    def __init__(
        self,
        vllm_client: vLLMClient,
        retriever: DenseRetriever,
        max_steps: int = 3
    ):
        self.vllm_client = vllm_client
        self.retriever = retriever
        self.max_steps = max_steps
        
    def _llm_call(self, prompt: str) -> Tuple[str, float]:
        """Execute LLM call and measure latency"""
        start_time = time.time()
        response = self.vllm_client.generate(prompt)
        latency = time.time() - start_time
        return response, latency

    def no_retrieval_pipeline(self, query: str) -> Dict[str, Any]:
        """Direct generation without retrieval"""
        prompt = f"Question: {query}\nAnswer:"
        
        answer, llm_latency = self._llm_call(prompt)
        
        return {
            "strategy": RAGStrategy.NO_RETRIEVAL.value,
            "answer": answer,
            "llm_calls": 1,
            "llm_latency": llm_latency,
            "retrieval_latency": 0.0,
            "total_latency": llm_latency,
            "steps": 0
        }

    def single_step_pipeline(self, query: str) -> Dict[str, Any]:
        """Single-step retrieval RAG"""
        # Retrieval
        docs, retrieval_latency = self.retriever.retrieve(query)
        context = format_context(docs)
        
        # Generation
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer, llm_latency = self._llm_call(prompt)
        
        return {
            "strategy": RAGStrategy.SINGLE_STEP.value,
            "answer": answer,
            "llm_calls": 1,
            "llm_latency": llm_latency,
            "retrieval_latency": retrieval_latency,
            "total_latency": llm_latency + retrieval_latency,
            "steps": 1
        }

    def multi_step_pipeline(self, query: str) -> Dict[str, Any]:
        """Multi-step RAG using IR-CoT (Interleaving Retrieval and Chain-of-Thought)"""
        total_llm_latency = 0.0
        total_retrieval_latency = 0.0
        llm_calls = 0
        all_docs = []
        current_reasoning = ""
        
        for step in range(self.max_steps):
            # 1. Retrieval based on query and current reasoning
            search_query = f"{query} {current_reasoning}".strip()
            docs, r_lat = self.retriever.retrieve(search_query, k=2)
            total_retrieval_latency += r_lat
            all_docs.extend(docs)
            
            # 2. IR-CoT Step: Generate next reasoning step or final answer
            context = format_context(all_docs)
            prompt = f"""Context:
{context}

Question: {query}
Reasoning so far: {current_reasoning}

Based on the context, provide the next step of reasoning to answer the question. If you have enough information, provide the final answer starting with 'Final Answer:'.
"""
            step_output, l_lat = self._llm_call(prompt)
            total_llm_latency += l_lat
            llm_calls += 1
            
            if "Final Answer:" in step_output:
                answer = step_output.split("Final Answer:")[-1].strip()
                break
            else:
                current_reasoning += " " + step_output
                answer = step_output # Fallback if max steps reached
        
        return {
            "strategy": RAGStrategy.MULTI_STEP.value,
            "answer": answer,
            "llm_calls": llm_calls,
            "llm_latency": total_llm_latency,
            "retrieval_latency": total_retrieval_latency,
            "total_latency": total_llm_latency + total_retrieval_latency,
            "steps": step + 1
        }

    def run(self, query: str, dataset_type: str) -> Dict[str, Any]:
        """Route query based on dataset type and run appropriate pipeline"""
        start_time = time.time()
        
        if dataset_type == "no-retrieval":
            result = self.no_retrieval_pipeline(query)
        elif dataset_type == "single-step":
            result = self.single_step_pipeline(query)
        elif dataset_type == "multi-step":
            result = self.multi_step_pipeline(query)
        else:
            # Random fallback if type is unknown
            strategy = random.choice(list(RAGStrategy))
            if strategy == RAGStrategy.NO_RETRIEVAL:
                result = self.no_retrieval_pipeline(query)
            elif strategy == RAGStrategy.SINGLE_STEP:
                result = self.single_step_pipeline(query)
            else:
                result = self.multi_step_pipeline(query)
        
        # Ensure total_latency is measured from the very start of the run call
        result["end_to_end_latency"] = time.time() - start_time
        return result
