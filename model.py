import torch
import torch.nn as nn

class TextGenerator(nn.Module):
    """
    LSTM-based text generation model.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_size: Dimension of word embeddings
        hidden_size: Dimension of LSTM hidden state
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
        
        Returns:
            Output tensor of shape (batch_size, seq_length, vocab_size)
        """
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Test model initialization
    model = TextGenerator(vocab_size=10000, embed_size=128, hidden_size=256)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    sample_input = torch.randint(0, 10000, (2, 50))  # batch_size=2, seq_length=50
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
