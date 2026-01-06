"""
vLLM Backend Client Configuration
"""
from openai import OpenAI
from typing import Optional, List, Dict, Any


class vLLMClient:
    """Client for interacting with vLLM backend server"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize vLLM client
        
        Args:
            base_url: vLLM server URL
            api_key: API key (default "EMPTY" for local vLLM)
            model_name: Model name served by vLLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using vLLM backend
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        return response.choices[0].text.strip()
    
    def chat_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using chat completion API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        return response.choices[0].message.content.strip()
