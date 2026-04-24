"""
Slack bot that uses a fine-tuned LLM to respond to messages.
- Responds when @mentioned in any channel or DM
- Once active in a thread, responds to ALL subsequent messages in that thread
- Uses the full thread history as context for each response
"""

import os
import re
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

DEFAULT_BASE_MODEL = "microsoft/phi-2"

# Track which threads Bolb has been active in: set of (channel, thread_ts)
active_threads: set = set()

# How many messages back to use as context (change this to whatever you want)
CONTEXT_MESSAGES = 1


def get_base_model_name(model_dir: str) -> str:
    """Read the base model name saved during training, with a sensible fallback."""
    base_model_file = Path(model_dir) / "base_model.txt"
    if base_model_file.exists():
        return base_model_file.read_text().strip()
    print(f"Warning: base_model.txt not found in {model_dir}, falling back to {DEFAULT_BASE_MODEL}")
    return DEFAULT_BASE_MODEL


def load_model(model_dir: str = "models/bolb-llm"):
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer

    if model is None or tokenizer is None:
        print(f"Loading model from: {model_dir}")

        base_model_name = get_base_model_name(model_dir)
        print(f"Base model: {base_model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_dir)
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def fetch_thread_context(client, channel: str, thread_ts: str) -> str:
    """
    Fetch the last CONTEXT_MESSAGES messages in a thread and format them
    as conversation context. ## messages are excluded from the count.
    Returns a formatted string like:
        User: hey bolb
        Bolb: hey!
        User: how are you?
    """
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts)
        messages = result.get("messages", [])

        context_lines = []
        for msg in messages:
            text = extract_user_text(msg.get("text", "")).strip()

            # Skip ## messages and empty messages
            if not text or text.startswith("##"):
                continue

            if msg.get("bot_id"):
                context_lines.append(f"Bolb: {text}")
            else:
                context_lines.append(f"User: {text}")

        # Only keep the last CONTEXT_MESSAGES messages
        context_lines = context_lines[-CONTEXT_MESSAGES:]
        # print(context_lines)

        return "\n".join(context_lines)

    except Exception as e:
        print(f"Error fetching thread context: {e}")
        return ""


def generate_response(context: str, max_new_tokens: int = 120) -> str:
    """
    Generate a response using the fine-tuned LLM.

    Args:
        context: Full conversation history formatted as "User: ...\\nBolb: ..." etc.
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text
    """
    if model is None or tokenizer is None:
        return "Model not loaded. Please train the model first."

    try:
        # Append "Bolb:" at the end so the model knows it's its turn to speak
        prompt = f"{context}\nBolb:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                top_p=0.98,
                top_k=60,
                temperature=0.6,
                do_sample=True,
                repetition_penalty=1.07,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n")[0],  # Stop at newline so it can't write multiple turns
                ],
            )

        # Only decode the newly generated tokens, not the prompt/context
        input_length = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        return generated_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"


def extract_user_text(message_text: str) -> str:
    """Strip Slack user mention tags (format: <@UXXXXXXXX>) from the message"""
    return re.sub(r"<@[A-Z0-9]+>", "", message_text).strip()


def handle_response(client, channel: str, thread_ts: str, say, logger):
    """Fetch thread context and respond"""
    global CONTEXT_MESSAGES

    if model is None:
        load_model()

    context = fetch_thread_context(client, channel, thread_ts)

    if not context:
        return

    response = generate_response(context)

    if not response:
        say("I'm not sure what to say to that!", thread_ts=thread_ts)
        return

    last_bot_message = next((m["text"] for m in reversed(client.conversations_replies(channel=channel, ts=thread_ts)["messages"]) if m.get("bot_id")), None)
    last_user_message = next((m["text"] for m in reversed(client.conversations_replies(channel=channel, ts=thread_ts)["messages"]) if not m.get("bot_id") and not m.get("subtype")), None)

    # print(f"Last bot message: {last_bot_message}")
    # print(f"Last user message: {last_user_message}")
    # print(f"Response to be sent: {response}")

    if last_bot_message != response:
        # print("context used")
        say(response, thread_ts=thread_ts)
        CONTEXT_MESSAGES += 1  # Only use the last message as context for this retry
    else:
        print(f"Memory Wiped from >{last_user_message}< becuase of >{last_bot_message}<")
        response = generate_response(last_user_message)
        CONTEXT_MESSAGES = 1  # Only use the last message as context for this retry

        if not response:
            say("I'm not sure what to say to that!", thread_ts=thread_ts)
            return

        say(response, thread_ts=thread_ts)

    #print(f"Response actually sent: {response}\n")


@app.event("app_mention")
def handle_app_mention(body, client, say, logger):
    """Handle @bolb mentions — mark the thread as active and respond"""
    try:
        event = body["event"]

        # Ignore messages starting with ##
        if extract_user_text(event.get("text", "")).startswith("##"):
            logger.info("Message starts with ##, ignoring.")
            return

        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]

        # Mark this thread as one Bolb is active in
        active_threads.add((channel, thread_ts))
        logger.info(f"Now active in thread {thread_ts} in {channel}")

        handle_response(client, channel, thread_ts, say, logger)

    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say(f"Sorry, I encountered an error: {str(e)}")


@app.event("message")
def handle_message(body, client, say, logger):
    """
    Handle all messages:
    - In DMs: always respond
    - In channels: respond if this is a thread Bolb is already active in,
                   and the message isn't from the bot itself
    """
    try:
        event = body["event"]

        # Ignore bot messages to avoid infinite loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Ignore messages starting with ##
        if extract_user_text(event.get("text", "")).startswith("##"):
            logger.info("Message starts with ##, ignoring.")
            return

        channel = event["channel"]
        channel_type = event.get("channel_type")
        thread_ts = event.get("thread_ts") or event["ts"]

        if channel_type == "im":
            # Always respond in DMs
            handle_response(client, channel, thread_ts, say, logger)

        elif (channel, thread_ts) in active_threads:
            # Respond to any new message in a thread where Bolb was mentioned
            handle_response(client, channel, thread_ts, say, logger)

    except Exception as e:
        logger.error(f"Error handling message: {e}")


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

    print(f"Starting Slack bot...")
    handler = SocketModeHandler(app, app_token)
    handler.start()


if __name__ == "__main__":
    main()