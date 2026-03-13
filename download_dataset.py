from dataset.data_collection import fetch_and_save
import os

print("="*60)
print("DOWNLOADING DATASET: Alice in Wonderland")
print("="*60)

# Download Alice in Wonderland from Project Gutenberg
try:
    # Try HTTPS first
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    fetch_and_save(url, "dataset.txt")
    print("\n✓ Successfully downloaded Alice in Wonderland from Project Gutenberg")
    
    # Check file size
    file_size = os.path.getsize("dataset.txt")
    print(f"✓ Dataset size: {file_size:,} bytes")
    
    # Show sample
    with open("dataset.txt", "r", encoding="utf-8") as f:
        sample = f.read(200)
    print(f"\nSample from dataset:\n{sample}...")
    
except Exception as e:
    print(f"\n✗ Error downloading from Gutenberg: {e}")
    print("\nAlternative: You can manually download from:")
    print("  https://www.gutenberg.org/files/11/11-0.txt")
    print("  and save it as 'dataset.txt' in the project root")

print("\n" + "="*60)
