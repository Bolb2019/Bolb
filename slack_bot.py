"""
Slack bot that uses a fine-tuned LLM to respond to messages.
Responds when messages contain #bolb mention in #bolbs-hideout channel.
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Global model and tokenizer
model = None
tokenizer = None


def load_model(model_dir: str = "models/bolb-llm"):
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print(f"Loading model from: {model_dir}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def generate_response(prompt: str, max_length: int = 150) -> str:
    """
    Generate a response using the fine-tuned LLM.
    
    Args:
        prompt: The input prompt
        max_length: Maximum length of generated text
        
    Returns:
        Generated text
    """
    if model is None or tokenizer is None:
        return "Model not loaded. Please train the model first."
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                top_p=0.92,
                top_k=50,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and return
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output (we only want the generated part)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"


@app.event("app_mention")
def handle_app_mention(body, say, logger):
    """Handle direct mentions of the bot"""
    try:
        message_text = body["event"]["text"]
        logger.info(f"App mention: {message_text}")
        
        if model is None:
            load_model()
        
        # Remove the bot mention from the text
        user_text = message_text.split(">", 1)[1].strip() if ">" in message_text else message_text
        
        response = generate_response(user_text)
        say(response)
    
    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say(f"Sorry, I encountered an error: {str(e)}")


def main():
    """Start the Slack bot"""
    # Pre-load the model
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("Model will be loaded on first use")
    
    # Start the bot using Socket Mode
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not app_token:
        print("Error: SLACK_APP_TOKEN not set in environment")
        return
    
    print("Starting Slack bot...")
    handler = SocketModeHandler(app, app_token)
    handler.start()


if __name__ == "__main__":
    main()
