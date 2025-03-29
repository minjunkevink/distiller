import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

class TextbookDataset(Dataset):
    """Dataset for training embedding models."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }
    
    def __len__(self):
        return len(self.encodings["input_ids"])

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_training_data() -> Tuple[List[str], List[str]]:
    """Load training data from splits directory."""
    training_dir = Path("data/training/splits")
    with open(training_dir / "train.json", "r") as f:
        data = json.load(f)
    
    # Extract input and output texts
    inputs = [item["input"] for item in data]
    outputs = [item["output"] for item in data]
    
    return inputs, outputs

def train_embedding_model():
    """Train a custom embedding model."""
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Base model for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load training data
    inputs, outputs = load_training_data()
    
    # Create datasets
    train_texts, val_texts = train_test_split(inputs, test_size=0.1, random_state=42)
    train_dataset = TextbookDataset(train_texts, tokenizer)
    val_dataset = TextbookDataset(val_texts, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Training loop
    num_epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Compute loss (using cosine similarity)
            loss = torch.nn.functional.cosine_embedding_loss(
                embeddings, embeddings,
                torch.ones(embeddings.size(0)).to(device)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                loss = torch.nn.functional.cosine_embedding_loss(
                    embeddings, embeddings,
                    torch.ones(embeddings.size(0)).to(device)
                )
                
                val_loss += loss.item()
        
        # Log metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = Path("outputs/models/embeddings")
            model_save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Saved best model to {model_save_path}")

if __name__ == "__main__":
    train_embedding_model() 