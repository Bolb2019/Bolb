# Bolb - AI-Powered Slack Bot

Your personal LLM-powered Slack bot that learns from your text and responds intelligently to `#bolb` mentions in #bolbs-hideout.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your training data
# Edit training_data.txt with content for the bot to learn from

# 3. Train the model
python train_llm.py training_data.txt

# 4. Configure Slack credentials
# Copy .env.example to .env and add your Slack tokens

# 5. Run the bot
python slack_bot.py
```

## Features

✨ **Fine-tuned LLM** - Trained on your custom text data
🚀 **Efficient** - Uses LoRA and 4-bit quantization to run on consumer hardware
💬 **Slack Integration** - Responds to #bolb mentions in real-time
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
- Slack app configuration
- Environment setup  
- Model training
- Troubleshooting

## Testing

```bash
# Test the model locally before deploying
python chat.py

# Run diagnostics to verify setup
python diagnostics.py
```

## How It Works

1. **Training Phase**: The script fine-tunes a base LLM (GPT-2) on your text data using LoRA adapters
2. **Inference Phase**: The Slack bot loads the trained model and generates responses when mentioned
3. **Efficiency**: Uses 4-bit quantization to fit large models on consumer GPUs

## Requirements

- Python 3.10+
- 8GB+ RAM (12GB recommended)
- GPU preferred (NVIDIA with CUDA)
- CPU-only is supported but much slower

## License

MIT

---

For more details, see [SETUP.md](SETUP.md)
an LLM trained off of myself
