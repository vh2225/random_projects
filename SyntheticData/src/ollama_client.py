"""
Ollama API Client

This module handles communication with the Ollama API.
It abstracts away the HTTP requests and provides a simple interface
for generating text using local LLMs.
"""

import requests
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    model: str = "llama3.2:8b-instruct-q4_K_M"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client
        
        Args:
            base_url: The base URL for Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
    def is_available(self) -> bool:
        """
        Check if Ollama is running and available
        
        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List[str]: List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, 
                 prompt: str, 
                 config: Optional[GenerationConfig] = None,
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            system_prompt: Optional system prompt for instructions
            
        Returns:
            str: Generated text
        """
        if config is None:
            config = GenerationConfig()
        
        # Build the full prompt with system message if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Prepare the request payload
        payload = {
            "model": config.model,
            "prompt": full_prompt,
            "stream": False,  # We'll use non-streaming for simplicity
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
            }
        }
        
        # Add max_tokens if specified
        if config.max_tokens:
            payload["options"]["num_predict"] = config.max_tokens
        
        try:
            # Send request to Ollama
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            
            # Extract the generated text
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error generating text: {e}")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             config: Optional[GenerationConfig] = None) -> str:
        """
        Chat completion using Ollama (better for conversation-style generation)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            config: Generation configuration
            
        Returns:
            str: Generated response
        """
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
            }
        }
        
        if config.max_tokens:
            payload["options"]["num_predict"] = config.max_tokens
        
        try:
            response = requests.post(self.chat_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in chat completion: {e}")