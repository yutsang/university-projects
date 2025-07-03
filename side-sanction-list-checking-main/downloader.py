import argparse
import logging
import sys
from datetime import date
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file(url: str, output_path: Path) -> None:
    try:
        logging.info(f"Downloading from {url} to {output_path}")
        response = urllib.request.urlopen(url)
        with open(output_path, "wb") as f:
            f.write(response.read())
        logging.info(f"Download completed: {output_path}")
    except Exception as e:
        logging.error(f"Failed to download file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL.")
    parser.add_argument('--url', type=str, required=False, default="https://scsanctions.un.org/consolidated/", help='URL to download from')
    parser.add_argument('--output-dir', type=str, required=False, default="Sanction_List", help='Directory to save the file')
    parser.add_argument('--filename', type=str, required=False, default=f"HKICPA_UN_Consolidated_list_{date.today()}.pdf", help='Output filename')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename

    download_file(args.url, output_path)

if __name__ == "__main__":
    main() 