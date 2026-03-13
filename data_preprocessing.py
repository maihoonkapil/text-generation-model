import nltk
import string
import os

# Download punkt tokenizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_data(file_path='dataset.txt'):
    """
    Load and preprocess text data.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        tokens: List of tokenized words
    """
    # Ensure we have the correct path
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(__file__), '..', file_path)
    
    # Load the raw text with UTF-8 encoding (FIX for Windows encoding issue)
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Tokenize into words
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Sample tokens: {tokens[:50]}")
    
    return tokens

if __name__ == "__main__":
    # Test preprocessing
    tokens = preprocess_data()
    print(f"\nPreprocessing complete. Total tokens: {len(tokens)}")
