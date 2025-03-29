import json
import os
from pathlib import Path
from typing import Dict, List
import yaml
from loguru import logger
import openai
from dotenv import load_dotenv
import time

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_training_data() -> List[Dict]:
    """Load training data from splits directory."""
    training_dir = Path("data/training/splits")
    with open(training_dir / "train.json", "r") as f:
        return json.load(f)

def format_training_data(data: List[Dict]) -> List[Dict]:
    """Format training data for OpenAI fine-tuning."""
    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in explaining textbook content."},
                {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"},
                {"role": "assistant", "content": item['output']}
            ]
        })
    return formatted_data

def create_fine_tuning_job():
    """Create and start a fine-tuning job with OpenAI."""
    # Load environment variables
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Load configuration
    config = load_config()
    
    # Load and format training data
    training_data = load_training_data()
    formatted_data = format_training_data(training_data)
    
    # Save formatted data to a temporary file
    temp_file = Path("data/training/temp_training.jsonl")
    with open(temp_file, "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
    
    try:
        # Upload the file
        with open(temp_file, "rb") as f:
            response = openai.File.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        
        # Create fine-tuning job
        job = openai.FineTuningJob.create(
            training_file=file_id,
            model=config["training"]["model"],
            hyperparameters={
                "n_epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate_multiplier": config["training"]["learning_rate"]
            }
        )
        
        logger.info(f"Created fine-tuning job: {job.id}")
        logger.info(f"Training file ID: {file_id}")
        
        # Monitor the job
        while True:
            job = openai.FineTuningJob.retrieve(job.id)
            status = job.status
            
            if status == "succeeded":
                logger.info("Fine-tuning completed successfully!")
                logger.info(f"Fine-tuned model: {job.fine_tuned_model}")
                break
            elif status == "failed":
                logger.error("Fine-tuning failed!")
                break
            elif status == "cancelled":
                logger.warning("Fine-tuning was cancelled!")
                break
            
            logger.info(f"Fine-tuning status: {status}")
            time.sleep(60)  # Check every minute
        
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()

if __name__ == "__main__":
    create_fine_tuning_job() 