import argparse
import logging
import sys
from pathlib import Path
from datetime import date
import subprocess
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_downloader(url: str, output_dir: str, filename: str):
    cmd = [
        sys.executable, 'downloader.py',
        '--url', url,
        '--output-dir', output_dir,
        '--filename', filename
    ]
    logging.info(f"Running downloader: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error("Downloader failed.")
        sys.exit(1)


def run_extractor(pdf_path: str, output_csv: Optional[str] = None):
    cmd = [sys.executable, 'extractor.py', '--pdf', pdf_path]
    if output_csv is not None:
        cmd += ['--output', output_csv]
    logging.info(f"Running extractor: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error("Extractor failed.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the sanction list download and extraction workflow.")
    parser.add_argument('--url', type=str, default="https://scsanctions.un.org/consolidated/", help='URL to download from')
    parser.add_argument('--output-dir', type=str, default="Sanction_List", help='Directory to save the file')
    parser.add_argument('--filename', type=str, default=f"HKICPA_UN_Consolidated_list_{date.today()}.pdf", help='Output filename for the PDF')
    parser.add_argument('--skip-download', action='store_true', help='Skip the download step')
    parser.add_argument('--skip-extract', action='store_true', help='Skip the extract step')
    parser.add_argument('--extract-output', type=str, help='Output CSV file path for extraction')
    args = parser.parse_args()

    pdf_path = str(Path(args.output_dir) / args.filename)

    if not args.skip_download:
        run_downloader(args.url, args.output_dir, args.filename)
    else:
        logging.info("Skipping download step as requested.")

    if not args.skip_extract:
        run_extractor(pdf_path, args.extract_output)
    else:
        logging.info("Skipping extract step as requested.")

if __name__ == "__main__":
    main() 