"""
Main synthetic data generator script
"""

import click
import json
import jsonlines
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm

from ollama_client import OllamaClient, GenerationConfig
from templates import TemplateManager
from utils import Config


class SyntheticDataGenerator:
    """Main class for generating synthetic data"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the generator with configuration"""
        self.config = Config(config_path)
        self.client = OllamaClient(base_url=self.config.get('ollama.base_url'))
        self.template_manager = TemplateManager()
        
        # Check if Ollama is available
        if not self.client.is_available():
            raise ConnectionError(
                "Ollama is not running. Please start Ollama with 'ollama serve'"
            )
    
    def generate_single(self, 
                       template_name: str, 
                       prompt_vars: Dict[str, Any],
                       model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a single data item
        
        Args:
            template_name: Name of the template to use
            prompt_vars: Variables to insert into the template
            model: Optional model override
            
        Returns:
            Generated data as dictionary
        """
        # Get the template
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Build the full prompt with examples
        few_shot_prompt = template.build_few_shot_prompt()
        user_prompt = template.format_prompt(**prompt_vars)
        
        full_prompt = f"{few_shot_prompt}\n\nNow generate:\n{user_prompt}\n\nOutput (JSON format):"
        
        # Create generation config
        gen_config = GenerationConfig(
            model=model or self.config.get('generation.model'),
            temperature=self.config.get('generation.temperature'),
            max_tokens=self.config.get('generation.max_tokens'),
            top_p=self.config.get('generation.top_p'),
            top_k=self.config.get('generation.top_k'),
            repeat_penalty=self.config.get('generation.repeat_penalty')
        )
        
        # Generate using chat format for better instruction following
        messages = [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.client.chat(messages, gen_config)
        
        # Try to parse the response as JSON
        try:
            # Extract JSON from the response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, return the raw response
            return {"raw_response": response, "error": "Failed to parse JSON"}
    
    def generate_batch(self,
                      template_name: str,
                      prompt_vars_list: List[Dict[str, Any]],
                      model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple data items
        
        Args:
            template_name: Name of the template to use
            prompt_vars_list: List of variable dictionaries
            model: Optional model override
            
        Returns:
            List of generated data items
        """
        results = []
        batch_size = self.config.get('batch.size', 10)
        batch_delay = self.config.get('batch.delay', 0.5)
        
        # Process in batches with progress bar
        with tqdm(total=len(prompt_vars_list), desc="Generating") as pbar:
            for i in range(0, len(prompt_vars_list), batch_size):
                batch = prompt_vars_list[i:i + batch_size]
                
                for vars_dict in batch:
                    try:
                        result = self.generate_single(template_name, vars_dict, model)
                        results.append(result)
                    except Exception as e:
                        print(f"\nError generating item: {e}")
                        results.append({"error": str(e), "vars": vars_dict})
                    
                    pbar.update(1)
                
                # Delay between batches to avoid overwhelming the system
                if i + batch_size < len(prompt_vars_list):
                    time.sleep(batch_delay)
        
        return results
    
    def save_results(self, 
                    results: List[Dict[str, Any]], 
                    output_format: str,
                    output_name: str) -> Path:
        """
        Save results to file
        
        Args:
            results: List of generated data
            output_format: Format (json, jsonl, csv)
            output_name: Base name for output file
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = Path(self.config.get('output.directory', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        # Add timestamp if configured
        if self.config.get('output.timestamp', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{output_name}_{timestamp}"
        
        # Save based on format
        if output_format == 'json':
            output_path = output_dir / f"{output_name}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif output_format == 'jsonl':
            output_path = output_dir / f"{output_name}.jsonl"
            with jsonlines.open(output_path, 'w') as writer:
                writer.write_all(results)
        
        elif output_format == 'csv':
            output_path = output_dir / f"{output_name}.csv"
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        return output_path


@click.command()
@click.option('--template', '-t', help='Template name to use')
@click.option('--count', '-c', default=1, help='Number of items to generate')
@click.option('--output-format', '-f', 
              type=click.Choice(['json', 'jsonl', 'csv']), 
              default='jsonl', 
              help='Output format')
@click.option('--output-name', '-o', default=None, help='Output file name (without extension)')
@click.option('--model', '-m', default=None, help='Model to use (overrides config)')
@click.option('--config', default=None, help='Path to config file')
@click.option('--list-templates', is_flag=True, help='List available templates')
@click.option('--list-models', is_flag=True, help='List available Ollama models')
def main(template, count, output_format, output_name, model, config, list_templates, list_models):
    """Generate synthetic data using Ollama and templates"""
    
    try:
        generator = SyntheticDataGenerator(config)
        
        # Handle list flags
        if list_templates:
            click.echo("Available templates:")
            for name in generator.template_manager.list_templates():
                tmpl = generator.template_manager.get_template(name)
                click.echo(f"  - {name}: {tmpl.description}")
            return
        
        if list_models:
            click.echo("Available Ollama models:")
            for model_name in generator.client.list_models():
                click.echo(f"  - {model_name}")
            return
        
        # Check if template is provided for generation
        if not template:
            click.echo("Error: --template is required for generation", err=True)
            click.echo("Use --list-templates to see available templates", err=True)
            return
        
        # Validate template exists
        if template not in generator.template_manager.list_templates():
            click.echo(f"Error: Template '{template}' not found", err=True)
            click.echo("Use --list-templates to see available templates", err=True)
            return
        
        # Set default output name if not provided
        if output_name is None:
            output_name = f"{template}_synthetic_data"
        
        click.echo(f"Generating {count} items using template '{template}'...")
        
        # Prepare prompt variables based on template
        prompt_vars_list = []
        tmpl = generator.template_manager.get_template(template)
        
        # Simple example inputs for each template type
        if template == 'customer_reviews':
            products = [
                "Laptop Stand", "Wireless Mouse", "USB Hub", "Webcam", 
                "Keyboard", "Monitor", "Desk Lamp", "Phone Charger"
            ]
            for i in range(count):
                prompt_vars_list.append({
                    "product_name": products[i % len(products)]
                })
        
        elif template == 'qa_pairs':
            topics = [
                "Python programming", "Machine learning", "Web development",
                "Data science", "Cloud computing", "Cybersecurity"
            ]
            for i in range(count):
                prompt_vars_list.append({
                    "topic": topics[i % len(topics)]
                })
        
        elif template == 'structured_data':
            data_types = ["user_profile", "product_catalog", "transaction_record"]
            for i in range(count):
                prompt_vars_list.append({
                    "data_type": data_types[i % len(data_types)]
                })
        
        elif template == 'facebook_profiles':
            profile_types = [
                "young_professional", "college_student", "parent", "retiree", 
                "artist", "entrepreneur", "teacher", "healthcare_worker"
            ]
            for i in range(count):
                prompt_vars_list.append({
                    "profile_type": profile_types[i % len(profile_types)]
                })
        
        # Generate data
        results = generator.generate_batch(template, prompt_vars_list, model)
        
        # Save results
        output_path = generator.save_results(results, output_format, output_name)
        
        click.echo(f"\nGeneration complete!")
        click.echo(f"Generated {len(results)} items")
        click.echo(f"Saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()