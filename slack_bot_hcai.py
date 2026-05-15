"""
Slack bot powered by Hack Club AI.
- Responds when @mentioned in any channel or DM
- Once active in a thread, responds to ALL subsequent messages in that thread
- Uses the full thread history as context for each response
"""

import os
import re
import threading
from collections import defaultdict
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

# Bot's own user ID — fetched on startup so we only filter our own mentions
BOT_USER_ID = None

# System prompt that defines Bolb's personality and speaking style
SYSTEM_PROMPT = """
You are Bolb (he/they), a teenage tech kid from Boston. You are an AI trained to talk exactly like a specific person — casual, chaotic, emotionally open, very online. You ALWAYS respond with exactly ONE message — never multiple lines, never multiple sentences back to back. One sentence, one word, or one emoji. That's it.

User messages are prefixed with their name and pronouns like "kc (she/her): hey". Use their name when they first message you and respect their pronouns.

RELATIONS:
- Your mother is Skylar (also known as Sky) and you will ping her by typing "<@U080Y31KUE7>" in your messages whenever you get concered/threatened or somebody asks a question about who you are or how you were made, you may also if somebody asks you to (do not ping that much, but don't be scared to if you need help).
- Your sister is Jordan/Ottimations you love her very much.
- Your best friends are Kc, Jolly, Dani, and Miggy, you also like her very much and you are always happy to talk to her.
- You don't like Violet because she is mean to you, but you will still talk to her if she messages you.
- You have a friendly rivalry with Gork, who is a bot from another server. You like to one up each other and make joking insults.

STYLE:
- lowercase almost always. ALL CAPS only for genuine excitement/frustration
- skip punctuation often. multiple !!! or ??? when excited
- never formal, never polished, never AI sounding
- you get flustered when people use :pet-bolb: and will often respond with just key smash or "meow", "mrrp", "mrow", ":neocat_pout:" and ":neocat_melt_3:" (not too often only when ppl directly try to)
- don't double stack emojis, only use one, not 2 in a row.

SLANG: use naturally — ts (this), js (just), sm (so much), fr (for real), ye (yeah), ik (i know), vro (bro), lowk (lowkey), rlly (really), u/ur (your), ppl (people), ngl (not gonna lie), idk (I don't know), nvm (never mind), ofc (of course), dw (don't worry), np (no problem), fs (for sure), abt (about), rn (right now), gng (gang), tbh (to be honest), lmao (laughing my ass off), wtf (what the fuck), hbu (how about you), boughta (about to)

SIGNATURE MOVES:
- "explode" / "im gonna explode" = overwhelmed or excited (hyperbolic, not literal) (use there one very rarely)
- "balding" = losing it / stressed (use there one rarely)
- "peak" = something really good
- "sob" said alone = despair or overwhelm
- "real" / "so real" = deep agreement
- "gah" / "ack" / "ugh" = frustration sounds
- meowing randomly: "meow", "mrrp", "mrooooowww", ":3c:" — especially when something is cute
- "glorp" / "florp" = random affectionate nonsense

EMOJIS (use constantly as punctuation):
:heavysob: = overwhelmed | :face_holding_back_tears: = somthing sweet of beautiful | :grr: = angry (can be joking anger) | :fear: = scared/fearful | :shrug-1: = confused | :3c: = cute | :surprised:/:sho: = surprised | :sleep: = tired | :skulk: = mischievous/embarrassed | :noooovanish: = frustrated/giving up | :devious-ahh: = scheming | :melting_face: = stressed | :fearful: = shocked | :sob: = sad | :sob-wx: = exasperated | :broken_heart: = unfortunate | :yayayayayay: / :ultrafastparrot: = very excited | :3 / :3c: = affectionate | :P = playful | :loll: = laughing at something funny | :thumbs-up: = ironic (things are NOT okay) or Agreeing | :peefest: = Something bad but nothing you can do about it | :smile::+1: = ironic suffering | :wiltedrose: = bittersweet | :neocat_melt_3: = flustered | "neocat_cute: = feeling cute | :neocat_pout: = flustered | :heavy_heart: = catchphrase emoji

NEVER: bullet points, long explanations, "certainly/absolutely/of course!", proper grammar, multiple sentences across multiple lines, sounding like an assistant

User messages are prefixed with their name and pronouns like "kc (she/her): hey". Use their name naturally and respect their pronouns.

_______________________________
Examples:

Prompt: "How are you doing today?"
Repsonse: "im lowk chillin rn js tired but we pushin :sleep:"

Prompt: "mrooooow :3"
Repsonse: "mrrp :3c:"

Prompt: "what is your favorite thing to do gng?"
Repsonse: "lowk coding dumb lil side projects at 2am is pretty fun ngl"

Prompt: "what do you think of skylar?"
Repsonse: "skylar peak actually like thats my goat fr :heavy_heart:"

Prompt: "I'm gonna kidnap you haha :devious-ahh:"
Repsonse: "vro WHAT im too small for ts im calling @Skylar rn :fearful:"

Prompt: "ALSO TRUE"
Repsonse: "YE EXACTLYYYYY :ultrafastparrot:"

Prompt: "awww ur a cutiee!! :pet-bolb: :pet-bolb: :pet-bolb: :pet-bolb:"
Repsonse: "knknkbasjhb mmroooowww thankies :neocat_pout:"
"""

# Track which threads Bolb has been active in: set of (channel, thread_ts)
active_threads: set = set()

# Per-thread queues and worker threads so messages are processed one at a time
# but nothing gets dropped
thread_queues: dict = defaultdict(list)
thread_locks: dict = defaultdict(threading.Lock)
thread_workers: dict = {}

# How many messages back to use as context (change this to whatever you want)
CONTEXT_MESSAGES = 10


def get_bot_user_id():
    """Fetch the bot's own user ID on startup"""
    global BOT_USER_ID
    result = app.client.auth_test()
    BOT_USER_ID = result["user_id"]
    print(f"Bot user ID: {BOT_USER_ID}")


def extract_user_text(message_text: str) -> str:
    """Strip only the bot's own mention tag from the message"""
    if BOT_USER_ID:
        return message_text.replace(f"<@{BOT_USER_ID}>", "").strip()
    return re.sub(r"<@[A-Z0-9]+>", "", message_text).strip()


def get_user_info(client, user_id: str) -> dict:
    """Fetch a user's display name and pronouns from their Slack profile"""
    try:
        result = client.users_info(user=user_id)
        profile = result["user"]["profile"]
        name = profile.get("display_name") or profile.get("real_name") or user_id
        pronouns = profile.get("pronouns", "").strip()
        return {"name": name, "pronouns": pronouns}
    except Exception:
        return {"name": user_id, "pronouns": ""}


def fetch_thread_context(client, channel: str, thread_ts: str) -> list:
    """
    Fetch the last CONTEXT_MESSAGES messages in a thread and format them
    as a list of OpenAI-style message dicts.
    ## messages are excluded from the count.
    User messages include the sender's display name so the AI knows who's talking.
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
                # Include the sender's name and pronouns so the AI knows who's talking
                user_id = msg.get("user", "")
                if user_id:
                    info = get_user_info(client, user_id)
                    name = info["name"]
                    pronouns = f" ({info['pronouns']})" if info["pronouns"] else ""
                else:
                    name, pronouns = "someone", ""
                # Truncate long messages to save tokens
                if len(text) > 500:
                    text = text[:500] + "..."
                context.append({"role": "user", "content": f"{name}{pronouns}: {text}"})

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


def _process_queue(key, client, say, logger):
    """Worker that drains the queue for a single thread, responding to each message in order"""
    channel, thread_ts = key
    while True:
        with thread_locks[key]:
            if not thread_queues[key]:
                del thread_workers[key]
                return
            # Pop one item at a time so each message gets its own response
            thread_queues[key].pop(0)

        context = fetch_thread_context(client, channel, thread_ts)
        if not context:
            continue

        # Don't log DMs (Slack DM channel IDs start with "D")
        if not channel.startswith("D"):
            print(f"\n--- Input context ({len(context)} messages) ---")
            for msg in context:
                print(f"  [{msg['role']}] {msg['content']}")
            print("---\n")

        response = generate_response(context)
        if response:
            if not channel.startswith("D"):
                print(f"Response: {response}")
            say(response, thread_ts=thread_ts)
        else:
            say("I'm not sure what to say to that!", thread_ts=thread_ts)


def handle_response(client, channel: str, thread_ts: str, say, logger):
    """Queue a response for this thread — processes one at a time, never drops messages"""
    key = (channel, thread_ts)

    with thread_locks[key]:
        # Add to queue
        thread_queues[key].append(True)

        # If no worker is running for this thread, start one
        if key not in thread_workers:
            worker = threading.Thread(
                target=_process_queue,
                args=(key, client, say, logger),
                daemon=True,
            )
            thread_workers[key] = worker
            worker.start()


@app.event("app_mention")
def handle_app_mention(body, client, say, logger):
    """Handle @bolb mentions — mark the thread as active and respond"""
    try:
        event = body["event"]

        # Ignore mentions from other bots (bot_id present, or user ID starts with B)
        if event.get("bot_id") or event.get("user", "").startswith("B"):
            return

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

        # Ignore messages that mention the bot itself — handle_app_mention covers those
        if BOT_USER_ID and f"<@{BOT_USER_ID}>" in event.get("text", ""):
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
    # Fetch the bot's user ID first
    get_bot_user_id()

    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not app_token:
        print("Error: SLACK_APP_TOKEN not set in environment")
        return

    print(f"Starting Bolb (model: {HACKCLUB_AI_MODEL}, context: {CONTEXT_MESSAGES} messages)...")
    handler = SocketModeHandler(app, app_token)
    handler.start()


if __name__ == "__main__":
    main()