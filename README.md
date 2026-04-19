# Bolb - AI-Powered Slack Bot

Your personal LLM-powered Slack bot that learns from your text and responds intelligently to `#bolb` mentions in #bolbs-hideout.

## Quick Start

```bash
# 1. Install dependencies
py -3.12 -m pip install -r requirements.txt

# 2. Prepare your training data
# Edit training_data.txt with content for the bot to learn from

# 3. Train the model
py -3.12 train_llm.py training_data.txt

# 4. Configure Slack credentials
# Copy .env.example to .env and add your Slack tokens

# 5. Run the bot
py -3.12 slack_bot.py
```

## Features

✨ **Fine-tuned LLM** - Trained on your custom text data using microsoft/phi-2
🚀 **Efficient** - Uses LoRA fine-tuning to run on consumer hardware
💬 **Slack Integration** - Responds to #bolb mentions in real-time
🎮 **GPU Accelerated** - Runs on NVIDIA GPUs with CUDA for fast responses
💰 **Free** - Uses open-source models, no API costs

## Files Overview

- `train_llm.py` - Fine-tune the LLM on your data
- `slack_bot.py` - Slack bot that responds to mentions
- `chat.py` - Interactive chat to test locally
- `diagnostics.py` - Verify your setup is correct
- `SETUP.md` - Detailed setup guide
- `training_data.txt` - Your training data (replace with your content)

## Setup Guide

See [SETUP.md](SETUP.md) for detailed instructions including:
- Python 3.12 installation
- CUDA/GPU setup
- Slack app configuration
- Model training
- Troubleshooting

## Testing

```bash
# Test the model locally before deploying
py -3.12 chat.py

# Run diagnostics to verify setup
py -3.12 diagnostics.py
```

## How It Works

1. **Training Phase**: The script fine-tunes microsoft/phi-2 (2.7B parameters) on your text data using LoRA adapters
2. **Inference Phase**: The Slack bot loads the trained model and generates responses when mentioned
3. **Efficiency**: Runs on your NVIDIA GPU via CUDA for fast inference

## Requirements

- Python 3.12 (3.14 is not supported by PyTorch)
- NVIDIA GPU with CUDA (CPU-only is very slow for phi-2)
- 8GB+ VRAM (RTX 3060 or better recommended)
- 16GB+ RAM

## License

MIT

---

For more details, see [SETUP.md](SETUP.md)