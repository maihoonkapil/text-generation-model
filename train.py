import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
import string
from collections import Counter
import numpy as np
import os
from src.model import TextGenerator

# Download NLTK data if needed
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_and_preprocess_data(file_path=None, seq_length=50):
    """Load text data, tokenize, create vocabulary and sequences."""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset.txt')
    
    # Load text with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Create vocabulary
    word_counts = Counter(tokens)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_size = len(vocab)
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Convert tokens to indices
    indexed_tokens = [word_to_idx[word] for word in tokens]
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(indexed_tokens) - seq_length):
        seq = indexed_tokens[i:i + seq_length]
        target = indexed_tokens[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets), vocab_size, word_to_idx, idx_to_word

def train_model_internal(model, train_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """Train the text generation model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # Get the last output for each sequence
            outputs = outputs[:, -1, :]  # Shape: (batch_size, vocab_size)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model

def train_model(seq_length=50, batch_size=32, embed_size=128, hidden_size=256, num_epochs=5, learning_rate=0.001):
    """
    Main training function called from main.py
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sequences, targets, vocab_size, word_to_idx, idx_to_word = load_and_preprocess_data(seq_length=seq_length)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of sequences: {len(sequences)}")
    
    # Create data loader
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.long),
                           torch.tensor(targets, dtype=torch.long))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = TextGenerator(vocab_size, embed_size, hidden_size)
    
    # Train model
    print("Training model...")
    trained_model = train_model_internal(model, train_loader, num_epochs, learning_rate, device)
    
    # Save model and vocabulary
    model_path = os.path.join(os.path.dirname(__file__), '..', 'text_generator.pth')
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'vocab.pth')
    
    torch.save(trained_model.state_dict(), model_path)
    torch.save({
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size,
        'seq_length': seq_length
    }, vocab_path)
    
    print(f"Training completed! Model saved as '{model_path}'")
    print(f"Vocabulary saved as '{vocab_path}'")

if __name__ == "__main__":
    train_model(num_epochs=5)
