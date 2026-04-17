"""
Quick setup and test script for Bolb.
Helps you get started and verify everything is working.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    
    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Checking Dependencies")
    
    required = [
        "torch",
        "transformers",
        "peft",
        "slack_bolt",
        "dotenv",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_slack_credentials():
    """Check if Slack credentials are configured"""
    print_section("Checking Slack Credentials")
    
    load_dotenv()
    
    app_token = os.environ.get("SLACK_APP_TOKEN")
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    
    if not app_token:
        print("❌ SLACK_APP_TOKEN not set in .env")
        print("   Get this from: https://api.slack.app.com/ → Your App → Settings → Socket Mode")
    else:
        print(f"✓ SLACK_APP_TOKEN found")
    
    if not bot_token:
        print("❌ SLACK_BOT_TOKEN not set in .env")
        print("   Get this from: https://api.slack.app.com/ → Your App → Settings → OAuth Tokens")
    else:
        print(f"✓ SLACK_BOT_TOKEN found")
    
    return app_token and bot_token


def check_training_data():
    """Check if training data exists"""
    print_section("Checking Training Data")
    
    if Path("training_data.txt").exists():
        size = Path("training_data.txt").stat().st_size
        print(f"✓ training_data.txt found ({size:,} bytes)")
        return True
    else:
        print("❌ training_data.txt not found")
        print("   Create one or replace the example file with your own data")
        return False


def check_model():
    """Check if model is trained"""
    print_section("Checking Trained Model")
    
    model_dir = Path("models/bolb-llm")
    
    if model_dir.exists():
        config_file = model_dir / "config.json"
        if config_file.exists():
            print(f"✓ Model found at {model_dir}")
            return True
        else:
            print(f"❌ Model directory exists but incomplete (missing config.json)")
            return False
    else:
        print(f"❌ Model not found at {model_dir}")
        print("   Train with: python train_llm.py training_data.txt")
        return False


def test_model_loading():
    """Test if the model can be loaded"""
    print_section("Testing Model Loading")
    
    try:
        print("Loading model (this may take a moment)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained("models/bolb-llm")
        model = AutoModelForCausalLM.from_pretrained(
            "models/bolb-llm",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print("✓ Model loaded successfully")
        print(f"  Device: {next(model.parameters()).device}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def run_diagnostics():
    """Run all diagnostic checks"""
    print("\n")
    print("╔────────────────────────────────────────────────────────╗")
    print("║           Bolb Setup Diagnostic Tool                   ║")
    print("╚────────────────────────────────────────────────────────╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Training Data", check_training_data),
        ("Slack Credentials", check_slack_credentials),
        ("Trained Model", check_model),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}\n")
    
    for name, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {name}")
    
    if passed == total:
        print("\n✓ All checks passed! Ready to run Slack bot with: python slack_bot.py")
        return True
    else:
        print("\n❌ Some checks failed. See above for details.")
        return False


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
