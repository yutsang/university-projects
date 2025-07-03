import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import load_and_clean_eta_data, get_regions
from map_routing import plot_shipment_route
import streamlit.components.v1 as components

st.title("ETA Prediction Dashboard")

# Load and clean data
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Present Materials/eta_prediction.csv'))
df = load_and_clean_eta_data(DATA_PATH)
regions = get_regions(df)

# Region selection
region = st.selectbox("Select Region", regions)

# Filter data by region
region_df = df[df['Loc'] == region]

st.write(f"### Shipments in {region}")
st.dataframe(region_df)

# Shipment selection
if not region_df.empty:
    shipment_idx = st.selectbox("Select a shipment to view route", region_df.index, format_func=lambda i: f"{region_df.loc[i, 'shipNo']} ({region_df.loc[i, 'origLoc']} → {region_df.loc[i, 'destLoc']})")
    shipment_row = region_df.loc[shipment_idx]
    # Assume region-specific graphml files are named as '<region>_speed.graphml' in Present Materials/
    graphml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../Present Materials/{region.lower()}_speed.graphml'))
    if os.path.exists(graphml_path):
        m = plot_shipment_route(shipment_row, graphml_path)
        m.save("temp_map.html")
        with open("temp_map.html", "r") as f:
            map_html = f.read()
        components.html(map_html, height=500)
    else:
        st.warning(f"GraphML file for region {region} not found: {graphml_path}")
else:
    st.info("No shipments available for this region.") 