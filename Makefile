# Makefile for Bolb LLM Slack Bot

.PHONY: help install train chat bot diagnostics clean

help:
	@echo "Bolb - LLM Slack Bot"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make train        - Train the LLM on training_data.txt"
	@echo "  make chat         - Test the model interactively"
	@echo "  make bot          - Run the Slack bot"
	@echo "  make diagnostics  - Run setup diagnostics"
	@echo "  make clean        - Remove cache and temporary files"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install"
	@echo "  2. Create/edit training_data.txt"
	@echo "  3. make train"
	@echo "  4. make chat      (optional: test locally)"
	@echo "  5. make diagnostics"
	@echo "  6. make bot"

install:
	@if not defined VIRTUAL_ENV ( \
		echo "Creating virtual environment..."; \
		python -m venv venv; \
		call venv\Scripts\activate.bat; \
	)
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

train:
	@if not exist "training_data.txt" ( \
		echo "Error: training_data.txt not found"; \
		exit /b 1; \
	)
	@echo "Starting model training..."
	python train_llm.py training_data.txt
	@echo "✓ Training complete"

chat:
	@echo "Starting interactive chat..."
	python chat.py

bot:
	@if not exist ".env" ( \
		echo "Error: .env file not found"; \
		echo "Copy .env.example to .env and fill in your Slack tokens"; \
		exit /b 1; \
	)
	@if not exist "models/bolb-llm" ( \
		echo "Error: Model not trained"; \
		echo "Run 'make train' first"; \
		exit /b 1; \
	)
	@echo "Starting Slack bot..."
	python slack_bot.py

diagnostics:
	@echo "Running setup diagnostics..."
	python diagnostics.py

clean:
	@echo "Cleaning up..."
	@for /d /r . %d in (__pycache__) do @if exist "%d" (rmdir /s /q "%d")
	@for /r . %f in (*.pyc) do @if exist "%f" (del "%f")
	@for /r . %f in (*.pyo) do @if exist "%f" (del "%f")
	@if exist ".pytest_cache" rmdir /s /q .pytest_cache
	@if exist "build" rmdir /s /q build
	@if exist "dist" rmdir /s /q dist
	@if exist "*.egg-info" rmdir /s /q *.egg-info
	@echo "✓ Cleaned up"
