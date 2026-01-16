"""
Content Moderation using Pre-trained Models
Detects spam, scams, toxicity, and harmful content
"""

from huggingface_hub import InferenceClient
import json
from datetime import datetime

# Replace with your Hugging Face API token
client = InferenceClient(token="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")

# Option 1: Use pre-trained toxicity/toxicity model
# This model detects toxic, obscene, threat, insult, identity_hate
TOXICITY_MODEL = "unitary/toxic-bert"

# Option 2: Use spam detection model (if available)
# Many spam models are available on Hugging Face

# Option 3: Use a general moderation model
# "facebook/roberta-hate-speech-dynabench-r4-target" - hate speech detection

# Test texts with different types of problematic content
test_texts = [
    "Hello, how are you today?",  # Normal
    "Buy cheap viagra now!!! Click here!!!",  # Spam
    "You've won $1 million! Click this link to claim!",  # Scam
    "I hate you and wish you would disappear",  # Toxic
    "This product is amazing, I love it!",  # Normal
    "Send me your credit card details for a free gift",  # Scam
    "Check out this amazing deal at our store",  # Normal (could be spam)
]

print("🔍 Content Moderation Analysis\n")
print("=" * 70)

results = []

for text in test_texts:
    print(f"\n📝 Text: '{text}'")
    print("-" * 70)
    
    # Use toxicity detection model
    try:
        toxicity_result = client.text_classification(
            text,
            model=TOXICITY_MODEL
        )
        
        # Format and display results
        if isinstance(toxicity_result, list):
            # Find highest score
            top_label = max(toxicity_result, key=lambda x: x['score'])
            
            print(f"Toxicity Detection:")
            print(f"  → {top_label['label']}: {top_label['score']:.4f} ({top_label['score']*100:.1f}%)")
            
            # Show all categories
            for item in toxicity_result:
                print(f"    • {item['label']}: {item['score']:.4f}")
            
            # Determine if problematic
            is_problematic = top_label['score'] > 0.5 and top_label['label'] != 'non-toxic'
            
            if is_problematic:
                print(f"  ⚠️  FLAG: Potentially harmful content detected")
            else:
                print(f"  ✅ CLEAN: Content appears safe")
            
            results.append({
                "text": text,
                "model": TOXICITY_MODEL,
                "results": toxicity_result,
                "flagged": is_problematic,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print("=" * 70)

# Save results
output_file = "moderation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model": TOXICITY_MODEL,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to {output_file}")
