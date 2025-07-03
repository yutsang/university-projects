# If you get an import error, run: pip install pandas
import pandas as pd
import argparse

def clean_data(file_path, output_path):
    """
    Load, clean, and save a CSV file.
    - Removes duplicates
    - Fills missing values (forward then backward fill)
    - Saves cleaned data to output_path
    """
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    data = data.ffill().bfill()
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and preprocess gold price data.")
    parser.add_argument('--input', type=str, default='gold_dec24(GC=F)_1d.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='cleaned_data.csv', help='Output cleaned CSV file path')
    args = parser.parse_args()
    clean_data(args.input, args.output)