# side-sanction-list-checking

Automate the workflow of comparing current customers to the UN consolidated sanction list to fulfill KYC/AML requirements in Hong Kong.

## Features
- Download the [UN Consolidated Sanction List](https://scsanctions.un.org/consolidated/) automatically
- Extract sanctioned persons and companies from the PDF into a CSV
- (Optional) Prepare for future Python-based client list comparison

## Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Download the Sanction List
```bash
python downloader.py --url "https://scsanctions.un.org/consolidated/" --output-dir Sanction_List --filename "HKICPA_UN_Consolidated_list_YYYYMMDD.pdf"
```
- All arguments are optional; defaults will be used if omitted.

### 2. Extract Names from the PDF
```bash
python extractor.py --pdf Sanction_List/HKICPA_UN_Consolidated_list_YYYYMMDD.pdf --output Sanction_List/HKICPA_UN_Consolidated_list_YYYYMMDD.csv
```
- The `--output` argument is optional; if omitted, a CSV with the same name as the PDF will be created.

## Project Structure
- `downloader.py`: Download sanction list PDF from the internet
- `extractor.py`: Extract names from PDF to CSV
- `requirements.txt`: Python dependencies
- `Sanction_List/`: Default directory for storing sanction lists

## Future Work
- Add Python-based client list comparison and fuzzy matching
- Integrate with Excel or other reporting tools

## References
- Fuzzy Lookup Algorithm: https://www.mrexcel.com/board/threads/fuzzy-matching-new-version-plus-explanation.195635/post-955137
- [UN Consolidated Sanction List](https://scsanctions.un.org/consolidated/)
