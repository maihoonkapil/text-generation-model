import torch
import os
from src.model import TextGenerator

def generate_text_internal(model, word_to_idx, idx_to_word, start_word, length=20, device='cpu'):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained TextGenerator model
        word_to_idx: Dictionary mapping words to indices
        idx_to_word: Dictionary mapping indices to words
        start_word: Starting word for generation
        length: Number of words to generate
        device: Device to run model on
    
    Returns:
        str: Generated text
    """
    model.eval()
    model.to(device)
    words = [start_word]
    
    for _ in range(length):
        # Convert current sequence to indices
        seq = [word_to_idx.get(word, 0) for word in words]  # Handle unknown words
        idx = torch.tensor([seq], dtype=torch.long).to(device)  # Shape: (1, len(words))
        
        # Get model prediction
        with torch.no_grad():
            output = model(idx)
        
        # Get the prediction for the last time step
        next_word_logits = output[0, -1, :]
        next_word_id = torch.argmax(next_word_logits).item()
        
        # Convert back to word
        next_word = idx_to_word[next_word_id]
        words.append(next_word)
    
    return " ".join(words)

def load_vocab(vocab_path=None):
    """Load vocabulary mappings from file."""
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(__file__), '..', 'vocab.pth')
    
    vocab_data = torch.load(vocab_path, weights_only=True)
    return vocab_data['word_to_idx'], vocab_data['idx_to_word'], vocab_data['vocab_size']

def load_model(model_path=None, vocab_size=10000, embed_size=128, hidden_size=256):
    """Load trained model from file."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'text_generator.pth')
    
    model = TextGenerator(vocab_size, embed_size, hidden_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def generate_text(start_words=None, length=50):
    """
    Main generation function called from main.py
    """
    if start_words is None:
        start_words = ["the", "artificial", "intelligence"]
    
    # Load vocab and model
    print("Loading vocabulary and model...")
    word_to_idx, idx_to_word, vocab_size = load_vocab()
    model = load_model(vocab_size=vocab_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate text
    print("\n" + "="*60)
    print("GENERATED TEXT SAMPLES")
    print("="*60)
    
    for start_word in start_words:
        if start_word.lower() not in word_to_idx:
            print(f"\nWarning: '{start_word}' not in vocabulary. Using 'the' instead.")
            start_word = "the"
        
        generated = generate_text_internal(model, word_to_idx, idx_to_word, start_word.lower(), length=length, device=device)
        print(f"\nStart word: '{start_word}'")
        print(f"Generated: {generated}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Example usage
    generate_text(start_words=["the", "artificial", "alice"], length=30)
