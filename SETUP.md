# Bolb - LLM-Powered Slack Bot

A Slack bot that learns from your custom text data and responds intelligently to messages containing `#bolb` mentions in the `#bolbs-hideout` channel.

## Features

- **Fine-tuned LLM**: Trained on your custom text file using LoRA (Low-Rank Adaptation) for efficient parameter tuning
- **Slack Integration**: Responds to `#bolb` mentions in real-time
- **Memory Efficient**: Uses 4-bit quantization to run on consumer hardware
- **Easy Setup**: Simple configuration with environment variables

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU (slower)
- ~16GB RAM (8GB minimum with quantization)

## Setup Instructions

### 1. Create Slack App

1. Go to https://api.slack.app.com/apps
2. Click "Create New App" → "From scratch"
3. Name it "Bolb" and select your workspace
4. Enable **Socket Mode** (Settings → Socket Mode → Toggle ON)
5. Generate an **App Token** (copy this to `.env` as `SLACK_APP_TOKEN`)

### 2. Configure Bot Permissions

1. Go to **OAuth & Permissions**
2. Add these scopes to **Bot Token Scopes**:
   - `chat:write`
   - `app_mentions:read`
3. Copy **Bot User OAuth Token** to `.env` as `SLACK_BOT_TOKEN`

### 3. Subscribe to Events

1. Go to **Event Subscriptions** → Toggle ON
2. Add these **Bot User Event Subscriptions**:
   - `app_mention`
   - `message.channels` (requires channel permission)

### 4. Install App to Workspace

1. Go to **Install App** → Click "Install to Workspace"
2. Authorize the app

### 5. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6. Create Your Training Data

Create a file named `training_data.txt` with the content you want your bot to learn from:

```bash
# Example:
echo "Your training text content here..." > training_data.txt
```

The file can contain:
- Chat logs
- Documentation
- Stories
- Articles
- Any text you want the bot to learn from

### 7. Train the Model

```bash
python train_llm.py training_data.txt
```

This will:
- Download a base LLM (GPT-2 by default)
- Fine-tune it on your text data using LoRA
- Save the trained model to `models/bolb-llm/`
- This takes 5-30 minutes depending on your hardware

### 8. Setup Environment Variables

```bash
# Copy the example and fill in your tokens
cp .env.example .env

# Edit .env with your actual Slack tokens:
SLACK_APP_TOKEN=xapp_your_token
SLACK_BOT_TOKEN=xoxb_your_token
```

### 9. Run the Bot

```bash
python slack_bot.py
```

The bot will:
- Load your trained model
- Connect to Slack via Socket Mode
- Listen for `#bolb` mentions
- Respond with AI-generated messages

## Usage

In your Slack channel `#bolbs-hideout`:

```
User: @Bolb #bolb tell me something interesting
Bot: [AI-generated response based on training data]
```

Or mention the bot directly:

```
User: @Bolb what do you think about this?
Bot: [AI-generated response]
```

## Advanced Configuration

### Use a Different Base Model

Edit `train_llm.py` to use a different model from Hugging Face:

```python
# Instead of "gpt2", try:
train_llm(data_file, model_name="distilgpt2")  # Smaller
train_llm(data_file, model_name="microsoft/phi-2")  # More powerful
```

### Adjust Training Parameters

In `train_llm.py`, modify:
- `num_train_epochs`: More = better but slower (default: 3)
- `per_device_train_batch_size`: Adjust if OOM (default: 4)
- `learning_rate`: Tweakfor convergence (default: 2e-4)

### Generate Longer Responses

In `slack_bot.py`, modify the `max_length` parameter in `generate_response()`:

```python
response = generate_response(user_text, max_length=300)  # Longer responses
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size in `train_llm.py`:
   ```python
   per_device_train_batch_size=2,
   ```

2. Use a smaller model:
   ```python
   train_llm(data_file, model_name="distilgpt2")
   ```

### Bot Not Responding

1. Check that `#bolbs-hideout` channel exists
2. Verify the bot has been added to the channel
3. Check logs: `python slack_bot.py` (watch for errors)
4. Ensure `.env` file has correct Slack tokens
5. Run `python train_llm.py` if you haven't already (model must exist)

### Model Loading Fails

1. Ensure `models/bolb-llm/` directory exists with model files
2. Run training: `python train_llm.py training_data.txt`
3. Check disk space (models ~2-4GB)

### Socket Mode Connection Issues

1. Verify `SLACK_APP_TOKEN` starts with `xapp_`
2. Check that Socket Mode is enabled in Slack App settings
3. Ensure app is installed to your workspace

## Project Structure

```
.
├── train_llm.py           # LLM training script
├── slack_bot.py           # Slack bot code
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your actual credentials (DON'T commit)
├── training_data.txt     # Your training data
└── models/
    └── bolb-llm/        # Fine-tuned model (generated after training)
```

## Performance Tips

- **First run**: Expect model download + training (~1-2 hours)
- **Inference**: Response takes 5-30 seconds depending on GPU
- **Memory**: ~8GB RAM minimum, 12GB+ recommended
- **GPU**: NVIDIA GPU highly recommended, CPU is very slow

## Cost

- 💰 Completely free! Uses open-source models and runs locally
- No API costs unlike cloud LLM services

## Future Enhancements

- [ ] Support for multiple channels
- [ ] Fine-tuning on Slack message history
- [ ] Custom response temperature/style settings
- [ ] Memory of past interactions
- [ ] Support for image generation
- [ ] Web dashboard for model monitoring

## License

MIT

## Support

For issues or questions, check the troubleshooting section or review the code comments.
