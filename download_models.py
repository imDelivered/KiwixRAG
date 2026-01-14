import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import config
from chatbot.model_manager import ModelManager

def main():
    print("=== Downloading Hermit AI Models ===")
    print("Pre-fetching models for offline use...")
    
    # Identify models from config
    models_to_download = [
        config.MODEL_QWEN_1_5B, # Fast Joint Model
        config.MODEL_QWEN_7B    # Smart/Reasoning Model
    ]

    # De-duplicate
    models_to_download = list(set(models_to_download))

    for model_id in models_to_download:
        print(f"\n[Checking] {model_id}...")
        try:
            # ensure_model_path handles checking local cache/file and downloading if needed
            path = ModelManager.ensure_model_path(model_id)
            print(f"✓ Ready: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}")
            sys.exit(1)

    print("\n✓ All models downloaded and ready for offline use.")

if __name__ == "__main__":
    main()
