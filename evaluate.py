import torch
import torch.nn as nn
import math
import os
from src.generate import load_vocab, load_model, generate_text_internal
from src.train import load_and_preprocess_data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def compute_perplexity(model, data_loader, criterion, device='cpu'):
    """
    Compute perplexity on the given data loader.
    
    Perplexity = exp(cross_entropy_loss)
    """
    model.eval()
    model.to(device)
    total_loss = 0
    total_words = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            # Get the last output for each sequence
            outputs = outputs[:, -1, :]  # Shape: (batch_size, vocab_size)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_words += targets.size(0)
    
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return perplexity

def evaluate_model(model_path=None, vocab_path=None, test_file=None, seq_length=50, batch_size=64):
    """
    Evaluate the trained model by computing perplexity and generating sample text.
    """
    # Set default paths
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'text_generator.pth')
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(__file__), '..', 'vocab.pth')
    if test_file is None:
        test_file = os.path.join(os.path.dirname(__file__), '..', 'dataset.txt')
    
    # Load vocab
    print("Loading vocabulary...")
    word_to_idx, idx_to_word, vocab_size = load_vocab(vocab_path)
    
    # Load model
    print("Loading model...")
    embed_size = 128  # Should match training
    hidden_size = 256
    model = load_model(model_path, vocab_size, embed_size, hidden_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    
    # Load and preprocess test data
    print("Loading test data...")
    sequences, targets, _, _, _ = load_and_preprocess_data(test_file, seq_length)
    
    # Use second half as test data
    split_idx = len(sequences) // 2
    test_sequences = sequences[split_idx:]
    test_targets = targets[split_idx:]
    
    print(f"Test set size: {len(test_sequences)} sequences")
    
    # Create test data loader
    test_dataset = TensorDataset(torch.tensor(test_sequences, dtype=torch.long),
                                 torch.tensor(test_targets, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute perplexity
    print("\nComputing perplexity...")
    criterion = nn.CrossEntropyLoss()
    perplexity = compute_perplexity(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print(f"MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Model Perplexity on test data: {perplexity:.2f}")
    print("(Lower perplexity = better model)")
    
    # Generate sample text
    print("\n" + "="*60)
    print("SAMPLE GENERATED TEXT")
    print("="*60)
    
    test_words = ["the", "alice", "i", "artificial", "intelligence"]
    for start_word in test_words:
        if start_word not in word_to_idx:
            continue
        generated = generate_text_internal(model, word_to_idx, idx_to_word, start_word, length=120, device=device)
        print(f"\nStart: '{start_word}'")
        print(f"Generated: {generated}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    evaluate_model()
