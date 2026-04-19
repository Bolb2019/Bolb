"""
Interactive test script to chat with your trained Bolb model locally.
Useful for testing before deploying to Slack.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import sys

DEFAULT_BASE_MODEL = "microsoft/phi-2"


def get_base_model_name(model_dir: str) -> str:
    """Read the base model name saved during training, with a sensible fallback."""
    base_model_file = Path(model_dir) / "base_model.txt"
    if base_model_file.exists():
        return base_model_file.read_text().strip()
    print(f"Warning: base_model.txt not found in {model_dir}, falling back to {DEFAULT_BASE_MODEL}")
    return DEFAULT_BASE_MODEL


def load_model(model_dir: str = "models/bolb-llm"):
    """Load the LoRA adapter on top of the base model"""
    try:
        print(f"Loading model from: {model_dir}")

        base_model_name = get_base_model_name(model_dir)
        print(f"Base model: {base_model_name}")

        # Load tokenizer from the adapter directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the base model, then apply the LoRA adapter on top
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()

        print("✓ Model loaded successfully\n")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


SYSTEM_PROMPT = "Bolb is a casual, witty person who gives short, natural responses.\n"


def generate_response(
    model,
    tokenizer,
    context: str,
    max_new_tokens: int = 60,
    temperature: float = 0.5,
    top_p: float = 0.92,
):
    """Generate a response from the model using full conversation context"""
    try:
        prompt = f"{SYSTEM_PROMPT}{context}\nBolb:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=40,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n\n")[0],
                ],
            )

        # Only decode the newly generated tokens, not the prompt/context
        input_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

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
    print("Type 'quit' or 'exit' to stop.")
    print("Messages starting with ## are ignored.\n")

    # Keep conversation history for context
    history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            # Ignore messages starting with ##
            if user_input.startswith("##"):
                print("(ignored)\n")
                continue

            # Build context from history
            history.append(f"User: {user_input}")
            context = "\n".join(history)

            print("\nBolb: ", end="", flush=True)
            response = generate_response(model, tokenizer, context=context)
            print(response)
            print()

            # Add Bolb's response to history for next turn
            history.append(f"Bolb: {response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()