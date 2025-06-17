"""
Template management for synthetic data generation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class Template:
    """Represents a single generation template"""
    
    def __init__(self, template_data: Dict[str, Any]):
        """Initialize template from dictionary"""
        self.name = template_data['name']
        self.description = template_data['description']
        self.system_prompt = template_data['system_prompt']
        self.few_shot_examples = template_data['few_shot_examples']
        self.user_prompt_template = template_data['user_prompt_template']
        self.output_schema = template_data.get('output_schema', {})
    
    def format_prompt(self, **kwargs) -> str:
        """
        Format the user prompt with provided variables
        
        Args:
            **kwargs: Variables to insert into template
            
        Returns:
            Formatted prompt string
        """
        return self.user_prompt_template.format(**kwargs)
    
    def build_few_shot_prompt(self) -> str:
        """Build a prompt with few-shot examples"""
        examples_text = []
        
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text.append(f"Example {i}:")
            if 'product' in example:
                examples_text.append(f"Product: {example['product']}")
            elif 'topic' in example:
                examples_text.append(f"Topic: {example['topic']}")
            elif 'data_type' in example:
                examples_text.append(f"Data Type: {example['data_type']}")
            
            examples_text.append(f"Output: {json.dumps(example['output'], indent=2)}")
            examples_text.append("")  # Empty line between examples
        
        return "\n".join(examples_text)


class TemplateManager:
    """Manages loading and accessing templates"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template manager
        
        Args:
            template_dir: Directory containing template files
        """
        if template_dir is None:
            base_dir = Path(__file__).parent.parent
            template_dir = base_dir / "templates"
        
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load all templates from directory"""
        templates = {}
        
        if not self.template_dir.exists():
            print(f"Warning: Template directory {self.template_dir} does not exist")
            return templates
        
        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                    template = Template(data)
                    templates[template.name] = template
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())