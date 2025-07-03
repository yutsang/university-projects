# side-sanction-list-checking

Automate the workflow of comparing current customers to the UN consolidated sanction list to fulfill KYC/AML requirements in Hong Kong.

## Features
- Download the [UN Consolidated Sanction List](https://scsanctions.un.org/consolidated/) automatically
- Extract sanctioned persons and companies from the PDF into a CSV
- (Optional) Prepare for future Python-based client list comparison
- **NEW:** Orchestrate the full workflow with a single command using `main.py`

## Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Orchestrate the Full Workflow
Run both download and extraction steps with one command:
```bash
python main.py
```

#### Options
- `--url`: URL to download from (default: UN list)
- `--output-dir`: Directory to save the PDF/CSV (default: Sanction_List)
- `--filename`: Output PDF filename (default: HKICPA_UN_Consolidated_list_YYYY-MM-DD.pdf)
- `--skip-download`: Skip the download step (useful if you already have the PDF)
- `--skip-extract`: Skip the extract step
- `--extract-output`: Output CSV file path (optional)

**Example:**
```bash
python main.py --url "https://scsanctions.un.org/consolidated/" --output-dir Sanction_List --filename "HKICPA_UN_Consolidated_list_20240601.pdf" --extract-output "Sanction_List/20240601.csv"
```

### 2. Download the Sanction List Only
```bash
python downloader.py --url "https://scsanctions.un.org/consolidated/" --output-dir Sanction_List --filename "HKICPA_UN_Consolidated_list_YYYYMMDD.pdf"
```
- All arguments are optional; defaults will be used if omitted.

### 3. Extract Names from the PDF Only
```bash
python extractor.py --pdf Sanction_List/HKICPA_UN_Consolidated_list_YYYYMMDD.pdf --output Sanction_List/HKICPA_UN_Consolidated_list_YYYYMMDD.csv
```
- The `--output` argument is optional; if omitted, a CSV with the same name as the PDF will be created.

## Project Structure
- `main.py`: Orchestrate the full workflow
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
