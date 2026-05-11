"""
Slack bot powered by Hack Club AI.
- Responds when @mentioned in any channel or DM
- Once active in a thread, responds to ALL subsequent messages in that thread
- Uses the full thread history as context for each response
"""

import os
import re
from openrouter import OpenRouter
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize Hack Club AI client
ai_client = OpenRouter(
    api_key=os.environ.get("HACKCLUB_AI_KEY", "dummy"),
    server_url="https://ai.hackclub.com/proxy/v1",
)

# Model to use
HACKCLUB_AI_MODEL = "openai/gpt-5.2-chat"

# System prompt that defines Bolb's personality and speaking style
SYSTEM_PROMPT = """You are Bolb, a casual and friendly Slack bot. keep your responses short and conversational, try to only use 1 sentence. You will use these messages as context and talk in a similar manner: 
"""

# Track which threads Bolb has been active in: set of (channel, thread_ts)
active_threads: set = set()

# How many messages back to use as context (change this to whatever you want)
CONTEXT_MESSAGES = 10


def fetch_thread_context(client, channel: str, thread_ts: str) -> list:
    """
    Fetch the last CONTEXT_MESSAGES messages in a thread and format them
    as a list of OpenAI-style message dicts.
    ## messages are excluded from the count.
    """
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts)
        messages = result.get("messages", [])

        context = []
        for msg in messages:
            text = extract_user_text(msg.get("text", "")).strip()

            # Skip ## messages and empty messages
            if not text or text.startswith("##"):
                continue

            if msg.get("bot_id"):
                context.append({"role": "assistant", "content": text})
            else:
                context.append({"role": "user", "content": text})

        # Only keep the last CONTEXT_MESSAGES messages
        return context[-CONTEXT_MESSAGES:]

    except Exception as e:
        print(f"Error fetching thread context: {e}")
        return []


def generate_response(context: list) -> str:
    """
    Generate a response using Hack Club AI.

    Args:
        context: List of OpenAI-style message dicts with role and content

    Returns:
        Generated text
    """
    try:
        response = ai_client.chat.send(
            model=HACKCLUB_AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *context,
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"


def extract_user_text(message_text: str) -> str:
    """Strip Slack user mention tags (format: <@UXXXXXXXX>) from the message"""
    return re.sub(r"<@[A-Z0-9]+>", "", message_text).strip()


def handle_response(client, channel: str, thread_ts: str, say, logger):
    """Fetch thread context and respond"""
    context = fetch_thread_context(client, channel, thread_ts)

    if not context:
        return

    response = generate_response(context)

    if not response:
        say("I'm not sure what to say to that!", thread_ts=thread_ts)
        return

    print(f"Response: {response}")
    say(response, thread_ts=thread_ts)


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
        
        if "@bolb" in extract_user_text(event.get("text", "")):
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
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not app_token:
        print("Error: SLACK_APP_TOKEN not set in environment")
        return

    print(f"Starting Bolb (model: {HACKCLUB_AI_MODEL}, context: {CONTEXT_MESSAGES} messages)...")
    handler = SocketModeHandler(app, app_token)
    handler.start()


if __name__ == "__main__":
    main()