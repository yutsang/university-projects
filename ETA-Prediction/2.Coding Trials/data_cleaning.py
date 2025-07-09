#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from numpy import datetime64
import numpy as np
import os

parent_directory = os.path.dirname(os.getcwd())
full_path = os.path.join(parent_directory, '1.Data', 'eta_prediction.csv')



pd.set_option('display.max_rows', None)
df = pd.read_csv(full_path)

df.columns = ["Loc", "Vehicle", "ShipNo", "Start-Lng", "Start-Lat", "Dest-Lng", "Dest-Lat", 
    "Time-1", "Time-2", "Time-3", "Time-4", "Time-5"]

for i in range(5):
    column = "Time-"+ str(i+1)
    df[column] = pd.to_datetime(df[column], errors = 'coerce', format='%Y-%m-%d %H:%M:%S')

convert_dict = {"Loc": str, "Vehicle": str, "ShipNo": str, "Start-Lng": str, "Start-Lat": str, "Dest-Lng": str, "Dest-Lat": str,
    "Time-5": datetime64, "Time-5": datetime64, "Time-5": datetime64, "Time-5": datetime64, "Time-5": datetime64}
df = df.astype(convert_dict)

df["planDurn"] = df["Time-2"].dropna()-df["Time-1"].dropna()
df["actDurn"] = df["Time-4"].dropna()-df["Time-3"].dropna()
df["planTaskTime"] = df["Time-5"].dropna()-df["Time-1"].dropna()
df["actTaskTime"] = df["Time-5"].dropna()-df["Time-3"].dropna()
#df["plannedRemainTime"] = df["Time-5"].dropna().astype(datetime64)-df["Time-2"].dropna().astype(datetime64)
#df["actualRemainTime"] = df["Time-5"].dropna().astype(datetime64)-df["Time-4"].dropna().astype(datetime64)

df['companyCode'] = df['ShipNo'].str[:3]
df['sysCode'] = df['ShipNo'].str[:7].str[4:]
df['countryCode'] = df['ShipNo'].str[:11].str[8:]
df['warehouse'] = df['ShipNo'].str[:16].str[12:]
df['shipNo'] = df['ShipNo'].str[-9:]



df = df.rename({"Time-1": "planStartTime", "Time-2": "planEndTime", "Time-3": "actStartTime", 
    "Time-4": "actEndTime", "Time-5": "MiscTime"}, axis = "columns")

df["origLoc"] = df[["Start-Lng", "Start-Lat"]].agg(', '.join, axis=1)
df["destLoc"] = df[["Dest-Lng", "Dest-Lat"]].agg(', '.join, axis=1)
df = df.drop(["Start-Lng", "Start-Lat", "Dest-Lng", "Dest-Lat", 'ShipNo'], axis=1)

cols = ["Loc", "Vehicle", "warehouse", "shipNo", "origLoc", "destLoc", "planStartTime", "planEndTime", "actStartTime", "actEndTime", 
    "MiscTime", "planDurn", "actDurn", "planTaskTime", "actTaskTime", "countryCode", "sysCode", "companyCode"]

df = df[cols]

df.head(15)


# In[8]:


#Current Strategy for Testing the Data
#data set = HKG, Date = 2019-01-02, Time = 11:40:55

df_hkg = df[df["Loc"] == "HKG"]
df_hkg["Date"] = df_hkg["planStartTime"].dt.date.astype(str)
df_hkg = df_hkg[df_hkg["Date"] == "2019-01-02"].drop({"Date"}, axis=1)
#df_hkg["Time"] = df_hkg["planStartTime"].dt.time.astype(str)
#df_hkg = df_hkg[df_hkg["Time"] == "11:40:55"].drop({"Time"}, axis=1).sort_values(by=["planStartTime"])
#df_hkg = df_hkg.drop_duplicates(subset=['destLoc'], keep="first", inplace=False)
df_hkg.index = np.arange(1, len(df_hkg) + 1)

#Assume only go to a dest once per journey
#1.Actual Start Time >= Planned Start Time
#2. Plan Duration > 0
#3. Actual Duration > 0
#4. Time-4 Not Null
#5. Check Time 1-3 Null? If null then drop
#6. Cal journey time by OSM, and set threshold for the actual Duration, intial 10 times

#df_hkg = df_hkg.drop_duplicates(subset=['Loc', 'Vehicle', 'origLoc', 'destLoc', 'planStartTime', 'planEndTime'], keep="first", inplace=False)
df_hkg.head(5)


# In[2]:


#primary_list: [columns] agg_col: columns to be aggregated to the new df
def drop_duplicate(dataframe, agg_col:str, primary_lists:list):
    ascend_boolean = []
    for item in primary_lists:
        ascend_boolean.append(True)
    dataframe.sort_values(primary_lists, axis=0, ascending=ascend_boolean, inplace=True, na_position="first")

    col_name = list(dataframe.columns.values)
    col_loc = col_name.index(agg_col)

    agg_col_cloned = agg_col+"_index"
    
    dataframe['is_duplicated'] = dataframe.duplicated(subset=primary_lists)
    
    index = 0
    dataframe["is_duplicated_index"] = 0
    shipNoDict = {}
    shipNoDictKey = []
    checker = False

    for idx, row in dataframe.iterrows():
        shipNoDictKey.append(row[agg_col])
        if row["is_duplicated"] == False:
            dataframe.loc[idx, "is_duplicated_index"] += index
            shipNoDict.update({index: shipNoDictKey})
            index += 1
            shipNoDictKey=[]
        elif row["is_duplicated"] == True:
            dataframe.loc[idx, "is_duplicated_index"] += -1
        
    dataframe = dataframe[dataframe["is_duplicated_index"]!=-1]
    dataframe[agg_col_cloned] = dataframe['is_duplicated_index'].map(shipNoDict)
    col_name[col_loc] = agg_col_cloned
    dataframe = dataframe[col_name]
    col_name[col_loc] = agg_col
    dataframe.rename({agg_col_cloned: agg_col})
    dataframe.reset_index(inplace=True)
    print(dataframe.columns)
    #dataframe = dataframe.drop(['is_duplicated', 'is_duplicated_index', 'index'], axis=1)
    dataframe = dataframe.drop(['index'], axis=1)

    return dataframe


# In[10]:


df_hkg = df[df["Loc"] == "HKG"]
df_hkg_cleansed = drop_duplicate(df_hkg, 'shipNo', ['Loc', 'Vehicle', 'origLoc', 'destLoc', 'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime'])
df_hkg_cleansed.to_csv('df_hkg_cleansed.csv')
df_hkg_cleansed.head(20)

