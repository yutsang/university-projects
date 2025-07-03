import pandas as pd
from numpy import datetime64

def load_and_clean_eta_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ["Loc", "Vehicle", "ShipNo", "Start-Lng", "Start-Lat", "Dest-Lng", "Dest-Lat", 
        "Time-1", "Time-2", "Time-3", "Time-4", "Time-5"]
    for i in range(5):
        column = "Time-"+ str(i+1)
        df[column] = pd.to_datetime(df[column], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df['companyCode'] = df['ShipNo'].str[:3]
    df['sysCode'] = df['ShipNo'].str[4:7]
    df['countryCode'] = df['ShipNo'].str[8:11]
    df['warehouse'] = df['ShipNo'].str[12:16]
    df['shipNo'] = df['ShipNo'].str[-9:]
    df['planDurn'] = (df['Time-2'] - df['Time-1']).dt.total_seconds() / 60
    df['actDurn'] = (df['Time-4'].fillna(df['Time-5']) - df['Time-3']).dt.total_seconds() / 60
    df['ActDurn-nonNull'] = (df['Time-4'].fillna(df['Time-5']) - df['Time-3']).dt.total_seconds() / 60
    df = df.rename({"Time-1": "planStartTime", "Time-2": "planEndTime", "Time-3": "actStartTime", 
        "Time-4": "actEndTime", "Time-5": "MiscTime"}, axis="columns")
    df["origLoc"] = df[["Start-Lng", "Start-Lat"]].agg(', '.join, axis=1)
    df["destLoc"] = df[["Dest-Lng", "Dest-Lat"]].agg(', '.join, axis=1)
    df = df.drop(["Start-Lng", "Start-Lat", "Dest-Lng", "Dest-Lat", 'ShipNo'], axis=1)
    cols = ["Loc", "Vehicle", "warehouse", "shipNo", "origLoc", "destLoc", "planStartTime", "planEndTime", "actStartTime", "actEndTime", 
        "MiscTime", "planDurn", "actDurn", "ActDurn-nonNull", "countryCode", "sysCode", "companyCode"]
    df = df[cols]
    df = df[df["actDurn"].notnull() & (df["ActDurn-nonNull"] >= 0)]
    return df

def get_regions(df):
    return sorted(df['Loc'].unique()) 