# Distiller 📚

> A powerful RAG pipeline fine-tuned for textbook PDF processing and information retrieval

## Overview 🎯

Distiller is an advanced Retrieval Augmented Generation (RAG) pipeline specifically designed for processing and querying textbook PDFs. It combines state-of-the-art language models with efficient information retrieval techniques to make textbook knowledge easily accessible.

### Pipeline Architecture
![RAG Pipeline](visualizations/pipeline.png)
*The Distiller RAG pipeline showing the complete flow from PDF input to interactive querying*

## Key Features ✨

- 📄 **Smart PDF Processing**: Extract clean text from complex textbook PDFs using PyMuPDF
- ✂️ **Intelligent Chunking**: Break down content into semantically meaningful chunks
- 🤖 **GPT-4 Summaries**: Generate concise, accurate summaries of textbook sections
- 🧮 **Vector Embeddings**: Create high-quality embeddings using OpenAI's latest models
- 🔎 **Fast Retrieval**: Efficient similarity search powered by FAISS indexing
- 💡 **Natural Queries**: Ask questions in plain language and get relevant answers
- 🎓 **Domain-Specific Fine-tuning**: Optimized for textbook content understanding

## System Architecture 🏗️
![Architecture Diagram](visualizations/architecture.png)
*Detailed system architecture showing the interaction between different components*

## Model Training & Fine-tuning 🎓

### Fine-tuning Process

1. **Data Collection**
   - Gather textbook PDFs from various subjects
   - Extract text and structure using PyMuPDF
   - Create question-answer pairs from textbook content
   - Generate summaries for different sections

2. **Data Preparation**
   ```python
   # Example data format
   {
       "instruction": "Explain the concept of...",
       "input": "Textbook section content...",
       "output": "Detailed explanation..."
   }
   ```

3. **Fine-tuning Options**

   a. **OpenAI Fine-tuning**
   ```bash
   # Prepare training data
   python src/prepare_training.py
   
   # Fine-tune GPT-4
   python src/fine_tune.py
   ```

   b. **Custom Model Training**
   ```bash
   # Train custom embedding model
   python src/train_embeddings.py
   
   # Train custom summarization model
   python src/train_summarizer.py
   ```

4. **Model Evaluation**
   - Accuracy on textbook-specific questions
   - Quality of generated summaries
   - Embedding quality metrics
   - Retrieval performance

### Training Data Requirements
- Minimum 1000 high-quality Q&A pairs
- Diverse subject coverage
- Various difficulty levels
- Balanced representation of concepts

### Fine-tuning Parameters
```yaml
# Fine-tuning Configuration
training:
  model: "gpt-4"
  epochs: 3
  batch_size: 4
  learning_rate: 1e-5
  warmup_steps: 100
  max_steps: 1000
  save_steps: 100
  eval_steps: 50

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_length: 2048
```

## Performance Metrics ⚡
![Benchmark Graph](visualizations/benchmark.png)
*Processing time benchmarks across different pipeline stages*

## Getting Started 🚀

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- GraphViz (for visualization)
- CUDA-capable GPU (for custom model training)

### Installation

1. **Using Conda**
```bash
conda create -n distiller python=3.9
conda activate distiller
pip install -r requirements.txt
```

2. **Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage Guide 📖

1. **Extract Text**
```bash
python src/extract.py
```

2. **Create Chunks**
```bash
python src/chunk.py
```

3. **Generate Summaries**
```bash
python src/summarize.py
```

4. **Create Embeddings**
```bash
python src/embeddings.py
```

5. **Query Information**
```bash
python src/retriever.py
```

## Project Structure 📁
```
distiller/
├── config/         # Configuration files
├── data/          # Input PDF files
│   ├── raw/       # Original PDFs
│   └── training/  # Training data
├── outputs/       # Generated outputs
├── src/           # Source code
│   ├── extract.py         # PDF text extraction
│   ├── chunk.py           # Text chunking
│   ├── summarize.py       # GPT-4 summarization
│   ├── embeddings.py      # Vector embeddings
│   ├── retriever.py       # Query interface
│   ├── visualize.py       # Visualization tools
│   ├── prepare_training.py # Training data preparation
│   ├── fine_tune.py       # OpenAI fine-tuning
│   ├── train_embeddings.py # Custom embedding training
│   └── train_summarizer.py # Custom summarizer training
├── tests/         # Test files
└── visualizations/# Generated diagrams
```

## Configuration ⚙️

Key settings in `config/config.yaml`:

```yaml
# Text Processing
chunk_size: 500
overlap: 50

# Models
summarization:
  model: "gpt-4"
  max_tokens: 1000

# Retrieval
faiss:
  metric: "cosine"
  k: 3

# Fine-tuning
training:
  model: "gpt-4"
  epochs: 3
  batch_size: 4
```

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap 🗺️

- [ ] Support for multiple PDF formats and layouts
- [ ] Advanced chunking algorithms with ML-based segmentation
- [ ] Additional embedding model options
- [ ] Interactive web interface
- [ ] Batch processing for multiple documents
- [ ] Enhanced visualization and analytics
- [ ] Multi-language support
- [ ] Automated fine-tuning pipeline
- [ ] Custom model training interface
- [ ] Distributed training support
