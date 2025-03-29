import json
import os
from pathlib import Path
from typing import List, Dict
import yaml
from loguru import logger
from tqdm import tqdm

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_qa_pairs(text: str, chunk_size: int = 500) -> List[Dict]:
    """Extract question-answer pairs from textbook content."""
    # Split text into chunks
    chunks = text.split("\n\n")
    qa_pairs = []
    
    for chunk in chunks:
        if len(chunk.split()) < chunk_size:
            continue
            
        # Generate questions based on content
        # This is a placeholder - in practice, you'd want to use a more sophisticated method
        questions = [
            f"What is the main concept discussed in this section?",
            f"Can you explain the key points in this section?",
            f"What are the important details about {chunk.split()[0]}?",
            f"How does this section relate to the overall topic?",
            f"What are the practical applications of this concept?"
        ]
        
        for question in questions:
            qa_pairs.append({
                "instruction": question,
                "input": chunk,
                "output": chunk  # In practice, you'd want to generate better answers
            })
    
    return qa_pairs

def prepare_training_data():
    """Prepare training data from textbook content."""
    config = load_config()
    data_dir = Path("data")
    training_dir = data_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    # Process each PDF in the data directory
    for pdf_file in data_dir.glob("raw/*.pdf"):
        logger.info(f"Processing {pdf_file}")
        
        # Extract text from PDF (using existing extract.py functionality)
        from extract import extract_text
        text = extract_text(str(pdf_file))
        
        # Generate Q&A pairs
        qa_pairs = extract_qa_pairs(text, config["chunk_size"])
        
        # Save training data
        output_file = training_dir / f"{pdf_file.stem}_training.json"
        with open(output_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {pdf_file.name}")

def split_training_data(train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split training data into train/val/test sets."""
    import random
    from pathlib import Path
    
    training_dir = Path("data/training")
    all_data = []
    
    # Load all training data
    for json_file in training_dir.glob("*_training.json"):
        with open(json_file, "r") as f:
            all_data.extend(json.load(f))
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Calculate split sizes
    n = len(all_data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split data
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save splits
    splits_dir = training_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(splits_dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    logger.info(f"Split data into {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")

if __name__ == "__main__":
    prepare_training_data()
    split_training_data() 