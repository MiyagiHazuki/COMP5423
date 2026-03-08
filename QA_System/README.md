# QA System

A comprehensive Question Answering System that combines multiple retrieval methods with large language models to provide accurate answers to user queries.

## Features

- **Hybrid Retrieval**: Combines BM25, TF-IDF, and deep learning-based retrieval methods
- **BGE-M3 Integration**: Supports advanced embedding models for semantic search
- **LLM Integration**: Uses Qwen LLM for answer generation
- **Web UI**: Interactive interface for querying the system
- **Comprehensive Evaluation**: Includes metrics calculation for system performance

## System Architecture

The QA system is composed of multiple modules:

1. **Retrieval Module**: Searches relevant documents using:
   - BM25/TF-IDF for keyword-based retrieval
   - DPR (Dense Passage Retrieval) with BGE-M3 embeddings
   - Hybrid approach combining multiple methods

2. **Generation Module**: Uses LLMs to generate natural language answers based on retrieved documents

3. **Evaluation Module**: Calculates metrics like recall and MRR to evaluate system performance

4. **Web UI**: Provides an interactive interface for users to query the system

## Installation

```bash

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The system is configured using `config.yaml`. Main configuration options include:

- Retrieval settings (methods, weights, top_k)
- Model paths and API keys
- Evaluation metrics
- Web UI settings

## Usage

### Running the QA System

```bash
python main.py --config config.yaml
```

### Command Line Arguments

- `--config`: Path to configuration file (default: config.yaml)
- `--verbose`: Enable verbose logging
- `--debug`: Run in debug mode

### Modes

The system supports multiple modes:

- **Normal mode**: Run the QA system with web UI
- **Process mode**: Process data and build necessary indices
- **Evaluation mode**: Run evaluation on test datasets
- **Vectorize mode**: Create vector embeddings for documents
- **Training mode**: Train DPR models

## Directory Structure

- `backbone/`: Core components for LLM integration
- `bm25/`: BM25 retrieval implementation
- `dpr/`: Dense Passage Retrieval implementation
- `faiss_composer/`: FAISS index management
- `models/`: Saved model files
- `data/`: Data storage
- `eval/`: Evaluation metrics and tools
- `utils/`: Utility functions
- `webui/`: Web interface components

## Example

1. Start the web UI:
   ```
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:8080`
3. Enter your question in the input box
4. Select the retrieval method (BM25, DPR, or Hybrid)
5. Click "Submit" to get your answer

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS for vector similarity search
- Transformers library
- Gradio for web UI