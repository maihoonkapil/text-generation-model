# text-generation-model

"# LSTM Text Generation Model

A PyTorch-based LSTM text generation model trained on \"Alice in Wonderland\" from Project Gutenberg.

## 🚀 Quick Start

### 1. Download Dataset
```bash
python download_dataset.py
```

### 2. Run Complete Pipeline
```bash
python main.py
```

This will:
- ✅ Preprocess the dataset
- ✅ Train the LSTM model
- ✅ Evaluate model performance (perplexity)
- ✅ Generate sample text

## 📁 Project Structure

```
.
├── main.py                          # Main pipeline
├── download_dataset.py              # Dataset downloader
├── dataset.txt                      # Training data (downloaded)
├── text_generator.pth               # Trained model (after training)
├── vocab.pth                        # Vocabulary mappings (after training)
├── requirements.txt                 # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Text preprocessing
│   ├── model.py                     # LSTM model definition
│   ├── train.py                     # Training logic
│   ├── generate.py                  # Text generation
│   └── evaluate.py                  # Model evaluation
└── dataset/
    ├── __init__.py
    └── data_collection.py           # Data fetching utilities
```

## 🔧 Individual Components

### Preprocess Data Only
```bash
python -c \"from src.data_preprocessing import preprocess_data; preprocess_data()\"
```

### Train Model Only
```bash
python -c \"from src.train import train_model; train_model(num_epochs=10)\"
```

### Generate Text Only (after training)
```bash
python -c \"from src.generate import generate_text; generate_text(['artificial', 'intelligence'], 50)\"
```

### Evaluate Model Only (after training)
```bash
python -c \"from src.evaluate import evaluate_model; evaluate_model()\"
```

## 🎯 Expected Output

After running `python main.py`, you should see:

```
STEP 4: GENERATING TEXT
============================================================

Start word: 'artificial'
Generated: artificial intelligence is rapidly changing the way humans interact...

Start word: 'intelligence'
Generated: intelligence is the ability to learn and adapt to new situations...

Start word: 'the'
Generated: the queen was in a great hurry and she was always ordering...
```

## ⚙️ Configuration

Edit hyperparameters in `src/train.py`:
- `seq_length`: Sequence length (default: 50)
- `batch_size`: Batch size (default: 32)
- `embed_size`: Embedding dimension (default: 128)
- `hidden_size`: LSTM hidden size (default: 256)
- `num_epochs`: Training epochs (default: 5)
- `learning_rate`: Learning rate (default: 0.001)

## 🐛 Fixed Issues

✅ **Encoding Error Fixed**: All file operations now use `encoding='utf-8'`
✅ **Import Issues Fixed**: Proper function exports and imports
✅ **Path Issues Fixed**: Relative paths work correctly
✅ **Module Structure Fixed**: Proper `__init__.py` files added

## 📊 Model Architecture

- **Input**: Sequence of word indices
- **Embedding Layer**: Maps words to dense vectors
- **LSTM Layer**: Captures sequential patterns
- **Output Layer**: Predicts next word probabilities

## 🎓 Training Details

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Evaluation Metric**: Perplexity (lower is better)

## 📝 Notes

- Training on CPU takes ~5-10 minutes with 5 epochs
- GPU training is much faster if available
- Model quality improves with more epochs (try 20-50)
- Larger hidden_size improves quality but increases training time
"
