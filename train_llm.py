"""
LLM Fine-tuning script using LoRA for efficient parameter tuning.
Trains a small model on custom text data.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
from pathlib import Path


def setup_quantization_config():
    """Setup quantization to reduce memory usage"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def setup_lora_config():
    """Setup LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        # GPT-2 uses 'c_attn' for attention layers
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text data"""
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def train_llm(
    data_file: str,
    model_name: str = "gpt2",
    output_dir: str = "models/bolb-llm",
    num_train_epochs: int = 3,
):
    """
    Fine-tune an LLM on custom text data using LoRA.
    
    Args:
        data_file: Path to text file to train on
        model_name: Hugging Face model name (default: gpt2 for lighter compute)
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
    """
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print("Model loaded and LoRA applied")
    
    # Prepare dataset using datasets library
    print(f"Loading training data from: {data_file}")
    
    # Read the text file
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into chunks and create dataset
    chunk_size = 512
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Create a Dataset from the chunks
    dataset = Dataset.from_dict({"text": chunks})
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=512),
        batched=True,
        remove_columns=["text"],
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else "training_data.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Training data file '{data_file}' not found")
        sys.exit(1)
    
    train_llm(data_file)
