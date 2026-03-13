import os
import argparse
import requests
from bs4 import BeautifulSoup

def fetch_and_save(url: str,
                   output_filename: str,
                   scrape: bool = False,
                   selectors: list = None) -> None:
    """
    Fetch content from *url* and write to ``output_filename``.
    
    If ``scrape`` is ``False`` (the default) the raw response text is saved.
    If ``scrape`` is ``True`` then ``selectors`` is used to find elements on the
    page and their text content is concatenated. This is useful when collecting
    text from HTML pages rather than plain text files like those on Gutenberg.
    """
    
    print(f"Fetching data from {url}...")
    response = requests.get(url, verify=False)  # Disable SSL verification for simplicity
    response.raise_for_status()
    
    if scrape:
        if selectors is None:
            selectors = ["p"]
        soup = BeautifulSoup(response.text, "html.parser")
        pieces = []
        for sel in selectors:
            for elem in soup.select(sel):
                pieces.append(elem.get_text())
        content = "\n".join(pieces)
    else:
        content = response.text
    
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Saved data from {url} to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download text data from a URL or scrape text from a webpage."
    )
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("output", help="Output filename to save data")
    parser.add_argument("--scrape", action="store_true",
                        help="Parse HTML and extract text using selectors")
    parser.add_argument("--selectors", nargs="+",
                        help="CSS selectors to extract when scraping")
    
    args = parser.parse_args()
    fetch_and_save(args.url, args.output, scrape=args.scrape,
                   selectors=args.selectors)
