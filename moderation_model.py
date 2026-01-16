"""
Interactive Content Moderation using Pre-trained Models
Detects spam, scams, toxicity, and harmful content
"""

from huggingface_hub import InferenceClient
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please create a .env file with your token.")

client = InferenceClient(token=HF_TOKEN)

TOXICITY_MODEL = "unitary/toxic-bert"
OUTPUT_FILE = "moderation_results.json"

# Load previous results if they exist
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data_store = json.load(f)
        results = data_store.get("results", [])
else:
    results = []

print("🔍 Interactive Content Moderation")
print("Type 'exit' to quit.\n")

while True:
    text = input("📝 Enter text to check: ").strip()
    
    if text.lower() in ["exit", "quit"]:
        break
    if not text:
        print("⚠️ Empty input, please type something.")
        continue

    try:
        # Run moderation model
        toxicity_result = client.text_classification(
            text,
            model=TOXICITY_MODEL
        )

        # Find highest scoring label
        top_label = max(toxicity_result, key=lambda x: x['score'])
        is_problematic = top_label['score'] > 0.5 and top_label['label'] != 'non-toxic'

        # Display results
        print("\n🎯 Moderation Results:")
        print(f"  → Top Label: {top_label['label']}")
        print(f"  → Confidence: {top_label['score']*100:.2f}%")
        if is_problematic:
            print("  ⚠️  FLAG: Potentially harmful content detected")
        else:
            print("  ✅ CLEAN: Content appears safe")

        # Optional: Show all labels and scores
        print("\n  • All Scores:")
        for item in toxicity_result:
            print(f"    - {item['label']}: {item['score']*100:.2f}%")

        # Save results
        results.append({
            "text": text,
            "model": TOXICITY_MODEL,
            "results": toxicity_result,
            "flagged": is_problematic,
            "timestamp": datetime.now().isoformat()
        })
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "model": TOXICITY_MODEL,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Result saved to {OUTPUT_FILE}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Error during moderation: {str(e)}")
        print("="*60 + "\n")