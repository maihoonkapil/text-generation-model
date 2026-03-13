
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.generate import generate_text
import os

print("="*60)
print("QUICK TEST - Text Generation Pipeline")
print("="*60)

# Check dataset
if not os.path.exists('dataset.txt'):
    print("❌ Error: dataset.txt not found!")
    print("Run: python download_dataset.py")
    exit(1)

# Test 1: Preprocessing
print("[1/4] Testing preprocessing...")
tokens = preprocess_data()
print(f"✓ Preprocessing works! Found {len(tokens)} tokens")

# Test 2: Quick training (1 epoch only for testing)
print("[2/4] Testing training (1 epoch, may take 2-3 minutes)...")
train_model(num_epochs=1, batch_size=64)
print("✓ Training works!")

# Test 3: Evaluation
print("[3/4] Testing evaluation...")
evaluate_model()
print("✓ Evaluation works!")

# Test 4: Generation
print("[4/4] Testing text generation...")
generate_text(start_words=["the", "alice"], length=20)
print("✓ Generation works!")

print(" " + "=*60")
print("ALL TESTS PASSED! ✓")
print("="*60)
print("To run full training, use: python main.py")
