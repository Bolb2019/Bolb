"""
Interactive test script to chat with your trained Bolb model locally.
Useful for testing before deploying to Slack.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import sys


def load_model(model_dir: str = "models/bolb-llm"):
    """Load the LoRA adapter on top of the base GPT-2 model"""
    try:
        print(f"Loading model from: {model_dir}")
        
        # Load tokenizer from the adapter directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the base GPT-2 model first, then apply the LoRA adapter on top
        base_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        
        print("✓ Model loaded successfully\n")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.92,
):
    """Generate a response from the model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    """Main chat loop"""
    model_dir = "models/bolb-llm"

    # Check if model directory exists
    if not Path(model_dir).exists():
        print(f"Error: Model not found at {model_dir}")
        print("Train the model first with: python train_llm.py training_data.txt")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(model_dir)

    print("=" * 60)
    print("Bolb Interactive Chat")
    print("=" * 60)
    print("Chat with your trained model locally!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            print("\nBolb: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()