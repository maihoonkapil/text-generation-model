from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.generate import generate_text
import os
import sys

def main():
    """
    Main entry point for the text generation pipeline.
    
    Steps:
    1. Preprocess dataset
    2. Train model
    3. Evaluate model
    4. Generate sample text
    """
    print("\n" + "="*60)
    print("TEXT GENERATION MODEL - COMPLETE PIPELINE")
    print("="*60 + "\n")
    
    # Check if dataset exists
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.txt')
    if not os.path.exists(dataset_path):
        print("✗ Error: dataset.txt not found!")
        print("\nPlease run: python download_dataset.py")
        print("Or manually download from: https://www.gutenberg.org/files/11/11-0.txt")
        sys.exit(1)
    
    try:
        # Step 1: Preprocessing dataset
        print("\n" + "="*60)
        print("STEP 1: PREPROCESSING DATASET")
        print("="*60)
        tokens = preprocess_data()
        print(f"✓ Preprocessing complete. Total tokens: {len(tokens)}")
        
        # Step 2: Training model
        print("\n" + "="*60)
        print("STEP 2: TRAINING MODEL")
        print("="*60)
        train_model(num_epochs=5, batch_size=32)  # Adjust epochs as needed
        print("✓ Training complete")
        
        # Step 3: Evaluating model
        print("\n" + "="*60)
        print("STEP 3: EVALUATING MODEL")
        print("="*60)
        evaluate_model()
        print("✓ Evaluation complete")
        
        # Step 4: Generating text
        print("\n" + "="*60)
        print("STEP 4: GENERATING TEXT")
        print("="*60)
        generate_text(start_words=["artificial", "intelligence", "the"], length=50)
        print("✓ Text generation complete")
        
        print("\n" + "="*60)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
