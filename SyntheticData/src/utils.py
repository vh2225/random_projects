"""
Configuration management for the synthetic data generator
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Manages configuration for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file (optional)
        """
        # Find the config file
        if config_path is None:
            # Look for default config.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration if file doesn't exist
            return self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'ollama': {
                'base_url': 'http://localhost:11434',
                'timeout': 120
            },
            'generation': {
                'model': 'llama3.2:8b-instruct-q4_K_M',
                'temperature': 0.7,
                'max_tokens': 500,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1
            },
            'output': {
                'directory': 'output',
                'format': 'jsonl',
                'timestamp': True
            },
            'batch': {
                'size': 10,
                'delay': 0.5
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'generation.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value