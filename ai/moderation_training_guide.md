# Training a Custom Moderation Model

## Pre-trained vs. Custom Training

### Option 1: Use Pre-trained Models (Recommended to Start) ✅

**Advantages:**
- ✅ No training needed - works immediately
- ✅ Already trained on large datasets
- ✅ Free to use via Hugging Face API
- ✅ Good for general spam/toxicity detection

**Available Models:**
1. **Toxicity Detection:**
   - `unitary/toxic-bert` - Detects toxic, obscene, threat, insult, identity_hate
   - `martin-ha/toxic-comment-model` - General toxicity

2. **Hate Speech:**
   - `facebook/roberta-hate-speech-dynabench-r4-target`
   - `cardiffnlp/twitter-roberta-hate-latest`

3. **Spam Detection:**
   - Search Hugging Face Hub: `spam-detection`
   - Many domain-specific models available

4. **Scam/Fraud Detection:**
   - Fewer pre-trained models available
   - May need custom training

**Usage Example:**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your-token")

# Check for toxicity
result = client.text_classification(
    "Your text here",
    model="unitary/toxic-bert"
)

# Check for spam (if model available)
result = client.text_classification(
    "Your text here",
    model="spam-detection-model-name"
)
```

### Option 2: Fine-tune Your Own Model (Custom Detection) 🎯

**When to Train:**
- ❌ Pre-trained models don't detect your specific spam patterns
- ❌ You have domain-specific scams (e.g., your industry's common scams)
- ❌ You have labeled data specific to your use case
- ❌ You need multi-label classification (spam + scam + toxicity together)

**Training Process:**

1. **Collect Training Data:**
   ```python
   TRAINING_DATA = [
       {"text": "Buy cheap pills now!", "labels": {"spam": 1, "scam": 0, "toxic": 0}},
       {"text": "You won $1000! Click here!", "labels": {"spam": 1, "scam": 1, "toxic": 0}},
       {"text": "I hate you", "labels": {"spam": 0, "scam": 0, "toxic": 1}},
       {"text": "Hello, how are you?", "labels": {"spam": 0, "scam": 0, "toxic": 0}},
   ]
   ```

2. **Choose a Base Model:**
   - `distilbert-base-uncased` - Fast, smaller
   - `roberta-base` - Better accuracy
   - `bert-base-uncased` - Balanced

3. **Training Approaches:**

   **Option A: Single Multi-label Model** (Detects all: spam, scam, toxic)
   ```python
   # Labels: [spam, scam, toxic] - binary for each
   # Example: [1, 0, 1] = spam and toxic, but not scam
   ```

   **Option B: Separate Models** (One for spam, one for scam, one for toxic)
   - Train 3 separate models
   - More flexible but requires 3x inference

4. **Training Script:**
   - Use `train_model.py` as a starting point
   - Modify for multi-label classification if needed
   - Consider using `AutoModelForSequenceClassification` with `num_labels=3` for multi-label

## Recommendation

**Start with Pre-trained Models:**
1. Use `unitary/toxic-bert` for toxicity
2. Try spam detection models from Hugging Face Hub
3. Evaluate if they meet your needs

**If Pre-trained Models Don't Work:**
1. Collect labeled data (spam/scam/toxic examples)
2. Use `train_model.py` and modify for your task
3. Fine-tune on your specific data
4. Consider using Google Colab for training (free GPU)

## Data Requirements for Training

**Minimum:**
- 100-500 examples per class (spam, scam, toxic, normal)
- Balanced dataset (equal examples of each)

**Recommended:**
- 1,000+ examples per class
- Diverse examples (different spam types, etc.)

## Example: Multi-label Classification Training

```python
# Labels format: [spam, scam, toxic]
# [1, 0, 0] = spam only
# [1, 1, 0] = spam and scam
# [0, 0, 1] = toxic only

TRAINING_DATA = [
    {"text": "Buy now!", "labels": [1, 0, 0]},
    {"text": "You won money!", "labels": [1, 1, 0]},
    {"text": "I hate you", "labels": [0, 0, 1]},
    {"text": "Hello friend", "labels": [0, 0, 0]},
]
```

## Resources

- **Hugging Face Model Hub:** https://huggingface.co/models?search=spam
- **Hugging Face Model Hub:** https://huggingface.co/models?search=toxicity
- **Google Colab:** https://colab.research.google.com (Free GPU for training)
