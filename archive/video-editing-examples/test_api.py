import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key (first 20 chars): {api_key[:20]}...")

client = anthropic.Anthropic(api_key=api_key)

# Test different models
models_to_test = [
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
]

print("\nTesting available models:\n")
for model in models_to_test:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ {model} - AVAILABLE")
    except anthropic.NotFoundError:
        print(f"❌ {model} - NOT FOUND")
    except Exception as e:
        print(f"❌ {model} - ERROR: {str(e)[:50]}")
