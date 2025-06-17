# Synthetic Data Generator

A flexible synthetic data generation tool using Ollama and local LLMs to create high-quality synthetic datasets for various use cases.

## Features

- 🚀 Fast local inference using Ollama
- 🎯 Multiple data generation templates (customer reviews, Q&A pairs, structured data)
- ⚙️ Configurable generation parameters
- 📊 Batch processing support
- 💾 Multiple output formats (JSON, CSV, JSONL)
- 🔧 Easy-to-extend template system

## Prerequisites

- Python 3.8+
- Ollama installed and running
- At least one LLM model installed in Ollama (recommended: llama3.2:8b-instruct-q4_K_M)

## Installation

1. Clone or create this project:
```bash
cd SyntheticData
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (if not already installed):
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama
```

4. Pull a model:
```bash
ollama pull llama3.2:8b-instruct-q4_K_M
```

## Quick Start

Generate synthetic customer reviews:
```bash
python src/generate.py --template customer_reviews --count 10
```

Generate Q&A pairs:
```bash
python src/generate.py --template qa_pairs --count 20 --output-format jsonl
```

## Project Structure

```
SyntheticData/
├── src/
│   ├── generate.py          # Main generation script
│   ├── ollama_client.py     # Ollama API wrapper
│   ├── templates.py         # Template management
│   └── utils.py            # Utility functions
├── templates/              # Prompt templates
│   ├── customer_reviews.json
│   ├── qa_pairs.json
│   └── structured_data.json
├── config/
│   └── config.yaml        # Configuration settings
├── output/               # Generated data output
└── requirements.txt      # Python dependencies
```

## Configuration

Edit `config/config.yaml` to customize:
- Model selection
- Temperature and generation parameters
- Output paths
- Batch sizes

## Templates

Templates define the structure and examples for synthetic data generation. Each template includes:
- System prompt
- Few-shot examples
- Output schema
- Validation rules

### Creating Custom Templates

1. Create a new JSON file in `templates/`
2. Define the prompt structure
3. Add examples
4. Reference it using `--template your_template_name`

## Examples

### Generate Product Descriptions
```bash
python src/generate.py --template product_descriptions --count 50 --model mistral:7b-instruct-q4_K_M
```

### Generate Training Data for Classification
```bash
python src/generate.py --template sentiment_classification --count 100 --output-format csv
```

### Batch Generation with Custom Config
```bash
python src/generate.py --config config/custom_config.yaml --template qa_pairs --count 1000
```

## Tips for Best Results

1. **Model Selection**: Larger models generally produce higher quality data but are slower
2. **Temperature**: Use 0.7-0.9 for creative tasks, 0.1-0.3 for structured data
3. **Few-shot Examples**: Provide 3-5 high-quality examples in your templates
4. **Validation**: Always validate generated data before using in production

## Troubleshooting

**Ollama not running:**
```bash
ollama serve
```

**Model not found:**
```bash
ollama list  # Check installed models
ollama pull model_name  # Install missing model
```

**Out of memory:**
- Use smaller quantized models (Q4_K_M instead of Q8)
- Reduce batch size in config
- Close other applications

## License

MIT License - feel free to use this for any purpose!