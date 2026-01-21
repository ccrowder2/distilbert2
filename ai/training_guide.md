# Training Guide: Fine-tuning Your Sentiment Analysis Model

## Overview
To train/fine-tune your sentiment analysis model, you need:
1. **Training Data** - Labeled examples of text and their sentiments
2. **Training Script** - Code to fine-tune the model
3. **Required Libraries** - Additional packages for training

## Step 1: Install Training Dependencies

```bash
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Install training libraries
pip install transformers datasets accelerate evaluate scikit-learn
```

## Step 2: Prepare Your Training Data

Your training data should be in this format:
```python
[
    {"text": "I love this!", "label": 2},      # POSITIVE
    {"text": "This is bad", "label": 0},       # NEGATIVE
    {"text": "It's okay", "label": 1},         # NEUTRAL
]
```

**Labels:**
- `0` = NEGATIVE
- `1` = NEUTRAL  
- `2` = POSITIVE

You can:
- Load from a JSON file
- Load from a CSV file
- Use Hugging Face datasets
- Manually create a list

## Step 3: Update the Training Script

Edit `train_model.py`:
1. Replace `TRAINING_DATA` with your actual dataset
2. Adjust `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS` as needed
3. Set `OUTPUT_DIR` to where you want to save the model

## Step 4: Run Training

```bash
python train_model.py
```

**Note:** Training requires significant resources. If PyTorch has DLL issues, consider:
- Using CPU (slower but works)
- Using Google Colab (free GPU)
- Using cloud services (AWS SageMaker, etc.)

## Alternative: Use Hugging Face AutoTrain

For easier training without coding:

```bash
pip install autotrain-advanced
autotrain app --help
```

## Step 5: Use Your Trained Model

After training, use your fine-tuned model:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./fine-tuned-model",  # Path to your trained model
    device=-1  # Use CPU
)

result = classifier("Your text here")
print(result)
```

## Tips

1. **More data = better model** - Aim for at least 100-1000 examples per class
2. **Balance your dataset** - Equal examples of each sentiment class
3. **Start small** - Test with a small dataset first
4. **Monitor training** - Watch for overfitting
5. **Save checkpoints** - The script saves model checkpoints during training

## Common Issues

- **Out of memory**: Reduce `BATCH_SIZE`
- **Training too slow**: Use GPU or cloud service
- **Poor accuracy**: Need more/better training data
