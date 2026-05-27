To test what a bot using this could look like you can chack out the one I made by mentioning <u>@bolb</u> in *#bolbs-testgrounds* (In hackclubs workplace) **Go test it by pinging it there**.

# General setup

There are two version you can make, you can either train a bot **entirely locally** or use **Hack club AI**.

General setup is for **BOTH** version, and each version will then have it's own individual instructions.

## Setup

Remove "template" from the name of _templace.env_ (make sure it's in the gitignore if you want to commit this to github)

You will need a slack bot, go to https://api.slack.com/apps and:
1. Create a new bot from scratch.
2. Name it something and add it to your workspace.
3. Enable _'Socket mode'_ and generate an app token, add this to the .env as **"SLACK_APP_TOKEN"**.
4. in the Oauth settings add _'chat:write'_ and _'app_mentions:read'_..
5. Add the _'Bot User OAuth Token'_ to the .env as **"SLACK_BOT_TOKEN"**.
6. in Event subscriptions add _'app\_mention'_ and _'message.channels'_.
7. Install the app to the workspace.

# Local llm

A local bot that responds to mentioning it and it should respond!

## Additional setup

1. PyTorch requires Python 3.12 or earlier, you can install it from here: https://www.python.org/downloads/release/python-3120/

2. Edit <u>_training\_data.txt_</u> to contain whatever content you want the AI to be trained off of *(the more data added the better the llm will turn out)*.

3. Run this to install requirements and train the llm
    ```bash
    py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    py -3.12 -m pip install -r requirements.txt
    py -3.12 train_llm.py training_data.txt
    ```

When installations and training finish the bot should be ready!

## Testing / running

Used to chat with the AI locally (Doesn't fully work, AI will behave differntly in slack, only use if you cannot use slack).
```bash
py -3.12 chat.py
```

Used as a simple check to make sure there are no major errors in the AI that was create, good to run right after training.
```bash
py -3.12 diagnostics.py
```

If all looks good, you can then run this to start the bot and messaging it on slack should have it able to respond!
```bash
py -3.12 slack_bot.py
```

## Requirements

- Python 3.12
- 8gb RAM
- 24gb storage
- GPU
- terminal access (whatever works, but scripts are in cmd)

# Hack club AI llm

Using Hack club AI makes the process a lot easier to set up. also works by just mentioning the bot and it should respond!

## Additional setup

1. Edit the **"SYSTEM_PROMPT"** of <u>_slack_bot_hcai.py_</u> to be whatever you want to bot to act as, the Prompt my bot uses is currently there as a sort of style guide / example.

2. Go to https://ai.hackclub.com/dashboard to create an account and an API key.
    - Add the API key to the .env as **"HACKCLUB_AI_KEY"**

3. Pick whatever AI model you would like to use and set it in **"HACKCLUB_AI_MODEL"** (by default it's GPT 5.2, but you can make it whatever you want).

## Testing / running

there isn't a whole lot in terms of testing outside of during runtime, so there is only one script to run the bot in slack.

```bash
python slack_bot_hcai.py #(or py -3.12 slack_bot_hcai.py)
```

## Requirements

- terminal access (whatever works, but scripts are in cmd)

## Disclosure of AI usage:

Claude Sonnet 4.6 was used to create a lot of this project but I did put in a lot of genuine work and probably put in more time than was tracked. but genAI was used to aid coding.