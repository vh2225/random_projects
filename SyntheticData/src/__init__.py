"""
Synthetic Data Generator Package
"""

from .ollama_client import OllamaClient, GenerationConfig
from .templates import Template, TemplateManager
from .utils import Config

__all__ = ['OllamaClient', 'GenerationConfig', 'Template', 'TemplateManager', 'Config']