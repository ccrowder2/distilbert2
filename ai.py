from huggingface_hub import InferenceClient
import json
from datetime import datetime

# Replace with your Hugging Face API token
client = InferenceClient(token="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")

# Test multiple strings to see how scores change
test_strings = [
    "I love using Hugging Face Transformers!",
    "a",
    "The weather is okay today.",
    "This is absolutely amazing!",
    "This product is terrible and broken."
]

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
results = []

print("Testing different strings and their sentiment scores:\n")
print("-" * 60)

for text in test_strings:
    result = client.text_classification(
        text,
        model=model_name
    )
    
    # Format result for display
    print(f"\nText: '{text}'")
    print(f"Result: {result}")
    
    # Find the highest scoring label
    if isinstance(result, list):
        top_result = max(result, key=lambda x: x['score'])
        print(f"Predicted: {top_result['label']} (score: {top_result['score']:.4f})")
    
    # Store for saving
    results.append({
        "text": text,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })
    print("-" * 60)

# Save results to file
output_file = "sentiment_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {output_file}")