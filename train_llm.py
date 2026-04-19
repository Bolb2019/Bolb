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

BASE_MODEL = "microsoft/phi-1_5"


def setup_lora_config():
    """Setup LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        # phi-2 uses standard projection layers for attention
        target_modules=["q_proj", "v_proj"],
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
    model_name: str = BASE_MODEL,
    output_dir: str = "models/bolb-llm",
    num_train_epochs: int = 3,
):
    """
    Fine-tune an LLM on custom text data using LoRA.

    Args:
        data_file: Path to text file to train on
        model_name: Hugging Face model name (default: microsoft/phi-2)
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
    """

    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Model loaded and LoRA applied")

    # Prepare dataset using datasets library
    print(f"Loading training data from: {data_file}")

    # Read the text file
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks and create dataset
    chunk_size = 512
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

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
        bf16=torch.cuda.is_bf16_supported(),  # prefer bfloat16 when available
        fp16=not torch.cuda.is_bf16_supported(),
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

    # Save the fine-tuned model and tokenizer
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the base model name so loaders know which base to reconstruct from
    with open(os.path.join(output_dir, "base_model.txt"), "w") as f:
        f.write(model_name)

    # Also save the base model's config.json so loaders can reconstruct
    # the architecture (PEFT only saves adapter files, not the base config)
    model.base_model.model.config.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    import sys

    data_file = sys.argv[1] if len(sys.argv) > 1 else "training_data.txt"

    if not os.path.exists(data_file):
        print(f"Error: Training data file '{data_file}' not found")
        sys.exit(1)

    train_llm(data_file)