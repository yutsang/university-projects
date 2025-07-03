import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import pdfplumber

def string_refining(test_string: str) -> str:
    output = test_string[:8] + test_string[13:]
    output = output.split(" ")
    for i in range(len(output)):
        output[i] = output[i].lstrip()
        for n in range(1, 7):
            output[i] = output[i].replace(f"{n}:", '')
        output[i] = output[i].replace(":", '').replace("na", '')
    output = list(filter(None, output))
    return ' '.join(output)

def extract_names_from_pdf(pdf_path: Path, output_csv: Path) -> None:
    try:
        logging.info(f"Extracting names from {pdf_path}")
        content = []
        idx = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                for line in lines:
                    if "Name:" in line and "(original script)" not in line:
                        content.insert(idx, line)
                        idx += 1
        sanction_list = pd.DataFrame()
        for item in content:
            refined = string_refining(item)
            sanction_list = sanction_list.append({'Code': refined[:7], 'Name': refined[8:]}, ignore_index=True)
        sanction_list.to_csv(output_csv, index=False)
        logging.info(f"Extraction completed: {output_csv}")
    except Exception as e:
        logging.error(f"Failed to extract names: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract names from a sanction list PDF and save as CSV.")
    parser.add_argument('--pdf', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--output', type=str, required=False, help='Output CSV file path')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logging.error(f"PDF file does not exist: {pdf_path}")
        sys.exit(1)
    output_csv = Path(args.output) if args.output else pdf_path.with_suffix('.csv')
    extract_names_from_pdf(pdf_path, output_csv)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 