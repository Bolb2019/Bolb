"""
Slack bot that uses a fine-tuned LLM to respond to messages.
Responds when messages contain #bolb mention in #bolbs-hideout channel.
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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

            base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, model_dir)
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def generate_response(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Generate a response using the fine-tuned LLM.

    Args:
        prompt: The input prompt
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text
    """
    if model is None or tokenizer is None:
        return "Model not loaded. Please train the model first."

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                top_p=0.92,
                top_k=50,
                temperature=0.7,
                do_sample=True,
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


import re

def extract_user_text(message_text: str) -> str:
    """Strip the @bolb mention (format: <@UXXXXXXXX>) from the message"""
    # Slack encodes mentions as <@USERID>, remove all of them
    return re.sub(r"<@[A-Z0-9]+>", "", message_text).strip()


def handle_mention(message_text: str, thread_ts: str, say, logger):
    """Shared logic for responding to any @bolb mention"""
    logger.info(f"Mention received: {message_text}")

    if model is None:
        load_model()

    user_text = extract_user_text(message_text)

    if not user_text:
        say("Hey! Mention me with a message and I'll respond.", thread_ts=thread_ts)
        return

    response = generate_response(user_text)

    if not response:
        say("I'm not sure what to say to that!", thread_ts=thread_ts)
        return

    say(response, thread_ts=thread_ts)


@app.event("app_mention")
def handle_app_mention(body, say, logger):
    """Handle @bolb mentions in channels"""
    try:
        event = body["event"]
        # Use existing thread if the mention is already in one, otherwise start a new thread
        thread_ts = event.get("thread_ts") or event["ts"]
        handle_mention(event["text"], thread_ts, say, logger)
    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say(f"Sorry, I encountered an error: {str(e)}")


@app.event("message")
def handle_direct_message(body, say, logger):
    """Handle @bolb mentions in DMs"""
    try:
        event = body["event"]
        # Only respond to DMs (channel_type: im), ignore bot messages
        if event.get("channel_type") == "im" and not event.get("bot_id"):
            thread_ts = event.get("thread_ts") or event["ts"]
            handle_mention(event.get("text", ""), thread_ts, say, logger)
    except Exception as e:
        logger.error(f"Error handling DM: {e}")


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