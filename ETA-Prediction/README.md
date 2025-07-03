# ETA Prediction

## Project Overview

This project predicts Estimated Time of Arrival (ETA) for logistics shipments using traffic, weather, and routing data. It demonstrates a full data science workflow: data cleaning, feature engineering, machine learning modeling, and interactive visualization.

## Workflow

1. **Data Preparation**
   - Clean and merge raw shipment, traffic (TomTom), and weather (NCEI) data.
   - Output: Processed CSV files for modeling.
   - Main notebook: `notebooks/0.data_clean+separate+cal_optimized_v2.ipynb`

2. **Model Training & Prediction**
   - Train Random Forest and XGBoost models to predict ETA based on historical data.
   - Output: Model files and prediction CSVs.
   - Main notebook: `notebooks/1.regression+modeling_v4.ipynb`

3. **Visualization**
   - Interactive dashboard for exploring predictions, routes, and shipment details by region.
   - **Current:** Streamlit app (`app/streamlit_app.py`)

## Directory Structure

- `app/` – Streamlit app and UI logic
- `src/` – Data processing and mapping modules
- `notebooks/` – Jupyter notebooks for EDA, prototyping, and modeling
- `Present Materials/` – Data files (csv, graphml, etc.)
- `requirements.txt` – Project dependencies
- `README.md` – Project overview and instructions

## Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Ensure `Present Materials/eta_prediction.csv` and region-specific GraphML files (e.g., `hkg_speed.graphml`, `sgp_speed.graphml`, etc.) are present in `Present Materials/`.

3. **Run the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Usage**
   - Select a region from the dropdown.
   - View and filter shipments for that region.
   - Select a shipment to view its route on an interactive map.

## Notes

- Large data files (e.g., TomTom speed, weather, and graphml) are required for full functionality.
- See `notebooks/` for detailed data processing and modeling steps.

## Authors

- HKUST IEDA FYP Group 3

---

**The project is now fully migrated to Streamlit for faster, multi-region ETA visualization.**
