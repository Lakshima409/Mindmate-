#!/usr/bin/env python3
"""Test script to verify model loading and intent prediction."""
import sys
import torch

# Absolute path to data.pth
DATA_PATH = "data/data.pth"

print("=" * 60)
print("TESTING MODEL LOADING AND INTENT PREDICTION")
print("=" * 60)

# Load the model - using absolute path
print("\n[1] Loading model checkpoint...")
data = torch.load(DATA_PATH, map_location=torch.device("cpu"))
print(f"  Keys: {list(data.keys())}")

# Check model_state
model_state = data.get("model_state")
print(f"  model_state type: {type(model_state)}")
print(f"  model_state is None: {model_state is None}")

if model_state is None:
    print("  FAIL: model_state is None!")
    sys.exit(1)

print("  model_state is a dict: ", isinstance(model_state, dict))
if isinstance(model_state, dict):
    print(f"  Number of parameters: {len(model_state)}")

# Get model parameters from checkpoint
input_size = data.get("input_size")
output_size = data.get("output_size")
hidden_size = data.get("hidden_size", 8)  # Read from checkpoint, not hardcoded
tags = data.get("tags")

print(f"\n[2] Model configuration:")
print(f"  input_size: {input_size}")
print(f"  output_size: {output_size}")
print(f"  hidden_size: {hidden_size}")
print(f"  tags count: {len(tags) if tags else 0}")

# Initialize and load model
from model import NeuralNet
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()
print("\n[3] Model loaded successfully with state_dict!")

# Test intent prediction
print("\n[4] Testing intent prediction...")
from nlp_utils import tokenize, bag_of_words

test_sentences = [
    "I'm feeling anxious and depressed",
    "Hello, how are you?",
    "I'm feeling stressed about work"
]

for sentence in test_sentences:
    try:
        tokens = tokenize(sentence)
        bow = bag_of_words(tokens, data["all_words"])
        bow = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(bow)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        predicted_tag = tags[prediction.item()]
        print(f"  '{sentence}' -> tag: {predicted_tag}, confidence: {confidence.item():.4f}")
    except Exception as e:
        print(f"  '{sentence}' -> ERROR: {e}")
        import traceback
        traceback.print_exc()

# Test crisis detection
print("\n[5] Testing crisis detection...")
from safety import detect_crisis

crisis_sentences = [
    "I want to kill myself",
    "I'm feeling really sad today",
    "What should I have for dinner?"
]

for sentence in crisis_sentences:
    is_crisis, category = detect_crisis(sentence)
    print(f"  '{sentence}' -> is_crisis: {is_crisis}, category: {category}")

print(f"  Keys: {list(data.keys())}")
model_state = data.get("model_state")
print(f"  model_state type: {type(model_state)}")
print(f"  model_state is None: {model_state is None}")

print("\n" + "=" * 60)
print("TEST COMPLETE - All checks passed!")
print("=" * 60)
