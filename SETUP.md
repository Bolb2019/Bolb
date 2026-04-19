# Bolb - LLM-Powered Slack Bot

A Slack bot that learns from your custom text data and responds intelligently to messages containing `#bolb` mentions in the `#bolbs-hideout` channel.

## Features

- **Fine-tuned LLM**: Trained on your custom text file using LoRA (Low-Rank Adaptation) with microsoft/phi-2 as the base model
- **Slack Integration**: Responds to `#bolb` mentions in real-time
- **GPU Accelerated**: Runs on NVIDIA GPUs with CUDA for fast training and inference
- **Easy Setup**: Simple configuration with environment variables

## Prerequisites

- **Python 3.12** (PyTorch does not support Python 3.13+ yet)
- **NVIDIA GPU** with CUDA support (RTX 3060 or better recommended)
- 16GB RAM recommended
- ~8GB free disk space for the model

## Setup Instructions

### 1. Install Python 3.12

PyTorch requires Python 3.12 or earlier. Download it from:
https://www.python.org/downloads/release/python-3120/

- Scroll down and grab **Windows installer (64-bit)**
- On the first install screen, **check "Add to PATH"**
- Verify the install:
```bash
py -3.12 --version
```

### 2. Install PyTorch with CUDA

Install PyTorch with CUDA 12.1 support for GPU acceleration:
```bash
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify your GPU is detected:
```bash
py -3.12 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name. If not, revisit this step before continuing.

### 3. Install Dependencies

```bash
py -3.12 -m pip install -r requirements.txt
```

### 4. Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name it "Bolb" and select your workspace
4. Enable **Socket Mode** (Settings → Socket Mode → Toggle ON)
5. Generate an **App Token** (copy this to `.env` as `SLACK_APP_TOKEN`)

### 5. Configure Bot Permissions

1. Go to **OAuth & Permissions**
2. Add these scopes to **Bot Token Scopes**:
   - `chat:write`
   - `app_mentions:read`
3. Copy **Bot User OAuth Token** to `.env` as `SLACK_BOT_TOKEN`

### 6. Subscribe to Events

1. Go to **Event Subscriptions** → Toggle ON
2. Add these **Bot User Event Subscriptions**:
   - `app_mention`
   - `message.channels`

### 7. Install App to Workspace

1. Go to **Install App** → Click "Install to Workspace"
2. Authorize the app

### 8. Create Your Training Data

Edit `training_data.txt` with the content you want your bot to learn from. The more text the better — aim for at least a few thousand words. It can contain:
- Chat logs
- Personal writing
- Articles or documentation
- Anything you want the bot to sound like

### 9. Train the Model

```bash
py -3.12 train_llm.py training_data.txt
```

This will:
- Download microsoft/phi-2 (~5GB, first run only)
- Fine-tune it on your text data using LoRA
- Save the trained adapter to `models/bolb-llm/`
- Training takes **20–60 minutes** on an NVIDIA GPU

You should see progress like:
```
Loading model: microsoft/phi-2
trainable params: 5,242,880 || all params: 2,784,926,720
Starting training...
{'loss': 2.34, 'step': 100}
...
Training complete!
```

### 10. Setup Environment Variables

Create a `.env` file in your project folder:
```
SLACK_APP_TOKEN=xapp-your-token-here
SLACK_BOT_TOKEN=xoxb-your-token-here
```

### 11. Run the Bot

```bash
py -3.12 slack_bot.py
```

The bot will load your trained model and connect to Slack via Socket Mode.

## Usage

In your Slack channel `#bolbs-hideout`:

```
User: @Bolb tell me something interesting
Bot: [AI-generated response based on your training data]
```

Or in a DM directly to the bot.

## Advanced Configuration

### Use a Different Base Model

Edit the `BASE_MODEL` line at the top of `train_llm.py`:

```python
BASE_MODEL = "microsoft/phi-2"        # Default — good balance of quality and speed
BASE_MODEL = "distilgpt2"             # Tiny, fast, CPU-friendly, lower quality
BASE_MODEL = "microsoft/phi-1_5"      # Lighter than phi-2, still good quality
```

The base model name is saved to `models/bolb-llm/base_model.txt` after training, so `chat.py` and `slack_bot.py` will automatically use the right one.

### Adjust Training Parameters

In `train_llm.py`, modify:
- `num_train_epochs`: More = better quality but slower (default: 3)
- `per_device_train_batch_size`: Reduce to 1 if you run out of VRAM (default: 2)
- `learning_rate`: Lower for more stable training (default: 2e-4)

### Generate Longer Responses

In `slack_bot.py`, modify the `max_new_tokens` parameter in `generate_response()`:

```python
response = generate_response(user_text, max_new_tokens=300)
```

## Testing Locally

Before deploying to Slack, test your trained model interactively:

```bash
py -3.12 chat.py
```

Run diagnostics to verify everything is set up correctly:

```bash
py -3.12 diagnostics.py
```

## Troubleshooting

### PyTorch not finding GPU

Make sure you installed the CUDA version of PyTorch:
```bash
py -3.12 -m pip uninstall torch -y
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory (OOM) during training

Reduce batch size in `train_llm.py`:
```python
per_device_train_batch_size=1,
```

Or switch to a smaller model like `microsoft/phi-1_5`.

### pip or python commands hanging

This usually means a zombie Python process is holding a lock. Restart your computer, then open a fresh command prompt and try again.

### Bot Not Responding

1. Check that `#bolbs-hideout` channel exists and the bot is added to it
2. Verify `.env` has correct Slack tokens
3. Make sure training completed and `models/bolb-llm/` exists
4. Check the terminal running `slack_bot.py` for error messages

### Socket Mode Connection Issues

1. Verify `SLACK_APP_TOKEN` starts with `xapp-`
2. Check that Socket Mode is enabled in your Slack App settings
3. Ensure the app is installed to your workspace

## Project Structure

```
.
├── train_llm.py           # LLM training script
├── slack_bot.py           # Slack bot
├── chat.py                # Local interactive chat for testing
├── diagnostics.py         # Setup verification tool
├── requirements.txt       # Python dependencies
├── .env                   # Your Slack credentials (don't commit this)
├── training_data.txt      # Your training data
└── models/
    └── bolb-llm/          # Fine-tuned LoRA adapter (generated after training)
        ├── base_model.txt # Records which base model was used
        └── ...
```

## Performance

- **Model download**: ~10 minutes (first run only, ~5GB)
- **Training**: 20–60 minutes on an NVIDIA GPU
- **Response time**: 5–15 seconds per message on GPU

## Cost

💰 Completely free — uses open-source models and runs entirely on your own hardware.

## License

MIT