#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from numpy import datetime64
import numpy as np
import os
import pandas as pd
#from sqlalchemy import null
from numpy import datetime64
import numpy as np
import matplotlib.pyplot as plt
import random
import folium
import osmnx as ox
import networkx as nx  
import difflib
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

#from folium.features import DivIcon

pd.set_option('display.max_rows', None)

hwy_speeds = {"residential": 10, "secondary": 33, "tertiary": 60}

dir = os.getcwd()
#Do not need to change the folder name
folder_name = "ETA-Prediction"

#Change filename for different regions or datasets
filename = "eta_prediction.csv"
#df_HKG.csv is a processed file that only extract HK data from the eta-prediction.csv

#path = dir[:dir.find(folder_name)]+folder_name+"-Data/1.Data/" + filename
path = filename

#from folium.features import DivIcon

pd.set_option('display.max_rows', None)

df_backup = pd.read_csv(path)
df = df_backup

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
df["Time-4"] = df["Time-4"].fillna(df["Time-5"])
df["ActDurn-nonNull"] = (df["Time-4"].dropna()-df["Time-3"].dropna()).dt.total_seconds()/60
#df["ActDurn-nonNull"] = df["actDurn-nonNull"].dt.total_seconds()
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
    "MiscTime", "planDurn", "actDurn", "ActDurn-nonNull", "countryCode", "sysCode", "companyCode"]

df = df[cols]

df.head(5)


# In[2]:


#basic cleaning #before (4965694, 17)
df = df[df["actDurn"].notnull() & (df["ActDurn-nonNull"] >= 0)]

startDate = df["planStartTime"].dt.strftime("%Y-%m-%d").min() #"2017-02-07"
endDate = df["planStartTime"].dt.strftime("%Y-%m-%d").max() #"2021-12-31"
#change these two strings

#df_hkg = df
df_hkg = df[df["Loc"] == "HKG"]
df_hkg = df_hkg[(df_hkg['planStartTime'] >= startDate) & (df_hkg['planStartTime'] <= endDate)]
#df_hkg.head() #(79578, 17)
#df_hkg = df

def primary_sort(df, columns_sort:list, duplicate_list:list, target_column: str):

    true_list = []

    for i in range(len(columns_sort)):
        true_list.append(True)

    df.sort_values(by=columns_sort, inplace=True, ascending = true_list)

    df['is_duplicated'] = df.duplicated(subset=duplicate_list)

    index = 0
    df["is_duplicated_index"] = 0
    shipNoDict = {}
    shipNoDictKey = []
    checker = False

    target_column_dummy = target_column+"_dummy"

    for idx, row in df.iterrows():
        shipNoDictKey.append(row[target_column])
        if row["is_duplicated"] == False:
            df.loc[idx, "is_duplicated_index"] += index
            shipNoDict.update({index: shipNoDictKey})
            index += 1
            shipNoDictKey=[]
        elif row["is_duplicated"] == True:
            df.loc[idx, "is_duplicated_index"] = -1
        
    result = df[df["is_duplicated_index"]!=-1]
    result[target_column_dummy] = result['is_duplicated_index'].map(shipNoDict)

    cols = result.columns
    '''cols = ['Loc', 'Vehicle', 'warehouse', 'ship_no', 'origLoc', 'destLoc',
       'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime',
       'MiscTime', 'planDurn', 'actDurn', 'ActDurn-nonNull', 'planTaskTime', 'actTaskTime',
       'countryCode', 'sysCode', 'companyCode', 'is_duplicated',
       'is_duplicated_index', 'shipNo']'''
    
    cols = list(map(lambda x: x.replace(target_column, target_column_dummy), cols))
    #cols = [(target_column_dummy if (target_column in s) else s) for s in cols]
    cols[-1] = target_column

    result = result[cols]
    result = result.drop(columns=cols[-3:])

    result = result.rename(columns={target_column_dummy: target_column})
    
    return result

sortList = ['Loc', 'Vehicle', 'warehouse', 'origLoc', 'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime']
duplicateList = ['Loc', 'Vehicle', 'warehouse', 'origLoc', 'destLoc', 'planStartTime', 'actStartTime', 'actEndTime']
targetCol = 'shipNo'
df_merge = primary_sort(df_hkg, sortList, duplicateList, targetCol)
df_merge.head()


# In[3]:


def primary_sort(df, columns_sort:list, duplicate_list:list, target_column: str):

    true_list = []

    for i in range(len(columns_sort)):
        true_list.append(True)

    df.sort_values(by=columns_sort, inplace=True, ascending = true_list)

    df['is_duplicated'] = df.duplicated(subset=duplicate_list)

    index = 0
    df["is_duplicated_index"] = 0
    shipNoDict = {}
    shipNoDictKey = []
    checker = False

    target_column_dummy = target_column+"_dummy"

    for idx, row in df.iterrows():
        shipNoDictKey.append(row[target_column])
        if row["is_duplicated"] == False:
            df.loc[idx, "is_duplicated_index"] += index
            shipNoDict.update({index: shipNoDictKey})
            index += 1
            shipNoDictKey=[]
        elif row["is_duplicated"] == True:
            df.loc[idx, "is_duplicated_index"] = -1
        
    result = df[df["is_duplicated_index"]!=-1]
    result[target_column_dummy] = result['is_duplicated_index'].map(shipNoDict)

    cols = result.columns
    '''cols = ['Loc', 'Vehicle', 'warehouse', 'ship_no', 'origLoc', 'destLoc',
       'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime',
       'MiscTime', 'planDurn', 'actDurn', 'ActDurn-nonNull', 'planTaskTime', 'actTaskTime',
       'countryCode', 'sysCode', 'companyCode', 'is_duplicated',
       'is_duplicated_index', 'shipNo']'''
    
    cols = list(map(lambda x: x.replace(target_column, target_column_dummy), cols))
    #cols = [(target_column_dummy if (target_column in s) else s) for s in cols]
    cols[-1] = target_column

    result = result[cols]
    result = result.drop(columns=cols[-3:])

    result = result.rename(columns={target_column_dummy: target_column})
    
    return result

sortList = ['Loc', 'Vehicle', 'warehouse', 'origLoc', 'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime']
duplicateList = ['Loc', 'Vehicle', 'warehouse', 'origLoc', 'destLoc', 'planStartTime', 'actStartTime', 'actEndTime']
targetCol = 'shipNo'
df_merge = primary_sort(df_hkg, sortList, duplicateList, targetCol)
df_merge.head()


# In[4]:


df_merge.shape


# In[5]:


#primary_list: [columns] agg_col: columns to be aggregated to the new df
def drop_duplicate(df, primary_lists:list, sort_only_col:str, agg_col:str):
    ascend_boolean = []
    sorting_list = primary_lists.remove(sort_only_col)
    print(sorting_list)
    for item in range(len(sorting_list)):
        ascend_boolean.append(True)
    df.sort_values(sorting_list, axis=0, ascending=ascend_boolean, inplace=True, na_position="first")
    col_name = list(df.columns.values)
    col_loc = col_name.index(agg_col)

    agg_col_cloned = agg_col+"_index"
    
    df['is_duplicated'] = df.duplicated(subset=primary_lists)
    
    index = 0
    df["is_duplicated_index"] = 0
    shipNoDict = {}
    shipNoDictKey = []
    checker = False

    for idx, row in df.iterrows():
        shipNoDictKey.append(row[agg_col])
        if row["is_duplicated"] == False:
            df.loc[idx, "is_duplicated_index"] += index
            shipNoDict.update({index: shipNoDictKey})
            index += 1
            shipNoDictKey=[]
        elif row["is_duplicated"] == True:
            df.loc[idx, "is_duplicated_index"] += -1
        
    df = df[df["is_duplicated_index"]!=-1]
    df[agg_col_cloned] = df['is_duplicated_index'].map(shipNoDict)
    #df = df.drop(["is_duplicated", "is_duplicated_index", agg_col], axis=1)
    col_name[col_loc] = agg_col_cloned
    df = df[col_name]

    return df


# In[6]:


#df_hkg_cleansed = drop_duplicate(df_hkg, ['Loc', 'Vehicle', 'warehouse', 'origLoc', 'destLoc', 'planStartTime', 'planEndTime', 'actStartTime', 'actEndTime'], 'destLoc', 'shipNo')
df_hkg_cleansed = df_hkg[df_hkg['is_duplicated_index']!=-1]

raw = df_hkg_cleansed.reset_index()
raw["sameRoute"] = raw.duplicated(subset=["Loc", "Vehicle", "warehouse", "origLoc", "planStartTime", "planEndTime", "actStartTime", "actEndTime"], keep="first")


# In[7]:


raw.shape


# In[8]:


dest = []
route = []
for index, row in raw.iterrows():
    if index < len(raw):
        if index == 0: 
            route.append(row["origLoc"])
            route.append(row["destLoc"])
        if row["sameRoute"] == False and index != 0: #start a new row
            #if route[0]!= route[-1]: route.append(route[0])
            dest.append(route)
            route = []
            route.append(row["origLoc"])
            route.append(row["destLoc"])
        if row["sameRoute"]: 
            if row["destLoc"]!= route[-1]: route.append(row["destLoc"])
    if index == len(raw) -1: 
        #route.append(route[0])
        dest.append(route)

df_dest = pd.DataFrame({"Route":dest}).reset_index()
raw_clean = raw[raw["sameRoute"]==False]
raw_clean = raw_clean.reset_index()
df_dest = pd.DataFrame({"Route":dest})
raw_clean["Route"] = df_dest

raw_clean = raw_clean[["Loc", "Vehicle", "warehouse", "shipNo", "Route", "planStartTime", "actStartTime",
                             "MiscTime", "planDurn", "actDurn", "ActDurn-nonNull"]]
raw_clean.head() #(2733, 11)


# In[9]:


raw_clean.shape


# In[10]:


#further clean -> believe in normal case journey time would be less than 5 days
raw_clean = raw_clean[raw_clean["ActDurn-nonNull"] < 5*24*60]
print(raw_clean.head()) #(1682, 11)
print(raw_clean.info(memory_usage='deep'))


# In[11]:


#test HKG only
raw_clean = raw_clean[raw_clean['Loc']=='HKG']
raw_clean.shape


# In[12]:


#===================================================================================================
#============================Direct Copy from main.ipynb below this line============================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================


# In[13]:


#def loadGraph(place="HK", optimizer="travel_time", mode = "drive", hwy_speeds=hwy_speeds):
    #ox.config(log_console = True, use_cache = True)
    ##mode = 'drive' # 'drive', 'bike', 'walk'
    #graph = ox.graph_from_place(place, network_type = mode)
    #graph = ox.add_edge_speeds(graph, hwy_speeds)
    #graph = ox.add_edge_travel_times(graph)
    #return graph

import json
import datetime
import osmnx as ox

#not in use
def loadGraph(graphml_file="hongkong_speed.graphml"):
    # load the street network and the saved edge speeds from the GraphML file
    ox.config(log_console = True, use_cache = True)
    #mode = 'drive' # 'drive', 'bike', 'walk'
    graph = ox.graph_from_place("HK", network_type = 'drive')
    # Define the file paths for the TomTom speed data
    rush_hour_path = './tomtom_speeds_weekday_rush_hours.json'
    non_rush_hour_path = './tomtom_speeds_weekday_non_rush_hours.json'

    # Define the start and end times for the two rush hour intervals
    rush_start_time_am = datetime.time(7, 0)   # Rush hour in the morning starts at 7:00 AM
    rush_end_time_am = datetime.time(10, 0)    # Rush hour in the morning ends at 10:00 AM
    rush_start_time_pm = datetime.time(16, 0)  # Rush hour in the afternoon starts at 4:00 PM
    rush_end_time_pm = datetime.time(19, 0)    # Rush hour in the afternoon ends at 7:00 PM

    # Define a dictionary to store the speed data as well as the rush hour classification
    speed_data = {}

    # Load the TomTom speed data into the dictionary for rush hour
    print('Loading', rush_hour_path)
    with open(rush_hour_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            start_time_str = data.get('startTime')
            if start_time_str:
                start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S').time()
                osm_id = data.get('frc', '') + str(data.get('id', ''))
                speed = data.get('currentSpeed', '')
                if osm_id and speed:
                    if rush_start_time_am <= start_time <= rush_end_time_am:
                        if osm_id not in speed_data:
                            speed_data[osm_id] = {'speed': speed, 'rush_hour': 'am'}
                        else:
                            speed_data[osm_id]['rush_hour'] = 'both'
                    elif rush_start_time_pm <= start_time <= rush_end_time_pm:
                        if osm_id not in speed_data:
                            speed_data[osm_id] = {'speed': speed, 'rush_hour': 'pm'}
                        else:
                            speed_data[osm_id]['rush_hour'] = 'both'

    # Load the TomTom speed data into the dictionary for non-rush hour
    print('Loading', non_rush_hour_path)
    with open(non_rush_hour_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            start_time_str = data.get('startTime')
            if start_time_str:
                osm_id = data.get('frc', '') + str(data.get('id', ''))
                speed = data.get('currentSpeed', '')
                if osm_id and speed:
                    if osm_id not in speed_data:
                        speed_data[osm_id] = {'speed': speed, 'rush_hour': 'none'}
                    else:
                        speed_data[osm_id]['rush_hour'] = 'none'

    # Load the street network from OpenStreetMap for Hong Kong
    print('Loading street network')
    #place_name = 'Hong Kong, China'
    #G = ox.graph_from_place(place_name, network_type='drive')

    # Assign the speed data to the edges in the street network
    print('Assigning speed data to edges')
    ox.speed.add_edge_speeds(graph, speed_data)
    ox.speed.add_edge_travel_times(graph)

    # add edge travel times to the graph
    #graph = ox.add_edge_travel_times(G)

    return graph

#not in use
#merge the existing two json files into one
def merge_speed_data(rush_hour_path, non_rush_hour_path):
    merged_speed_data = {}

    # Load the rush hour data
    with open(rush_hour_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            osm_id = data.get('osm_id')
            speed = data.get('currentSpeed')
            if osm_id and speed:
                if osm_id not in merged_speed_data:
                    merged_speed_data[osm_id] = [float(speed)]
                else:
                    merged_speed_data[osm_id].append(float(speed))

    # Load the non-rush hour data
    with open(non_rush_hour_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            osm_id = data.get('osm_id')
            speed = data.get('currentSpeed')
            if osm_id and speed:
                if osm_id not in merged_speed_data:
                    merged_speed_data[osm_id] = [float(speed)]
                else:
                    merged_speed_data[osm_id].append(float(speed))

    # Calculate the average speed for each osm_id
    average_speed_data = {}
    for osm_id, speeds in merged_speed_data.items():
        average_speed = sum(speeds) / len(speeds)
        average_speed_data[osm_id] = average_speed

    # Return the average speed data as a JSON string
    return json.dumps(average_speed_data)

#not in use
def loadGraph_v0(graphml_file="hongkong_speed.graphml"):
    # Define the file paths for the TomTom speed data
    rush_hour_path = './tomtom_speeds_weekday_rush_hours.json'
    non_rush_hour_path = './tomtom_speeds_weekday_non_rush_hours.json'

    # Define the start and end times for the two rush hour intervals
    rush_start_time_am = datetime.time(7, 0)   # Rush hour in the morning starts at 7:00 AM
    rush_end_time_am = datetime.time(10, 0)    # Rush hour in the morning ends at 10:00 AM
    rush_start_time_pm = datetime.time(17, 0)  # Rush hour in the evening starts at 5:00 PM
    rush_end_time_pm = datetime.time(20, 0)    # Rush hour in the evening ends at 8:00 PM

    # Load the TomTom speed data from the JSON files
    with open(rush_hour_path) as f:
        rush_hour_data = json.load(f)
    with open(non_rush_hour_path) as f:
        non_rush_hour_data = json.load(f)

    # Merge the rush hour and non-rush hour data into a single dictionary
    merged_speed_data = {**rush_hour_data, **non_rush_hour_data}

    # Create a dictionary to store the average speed for each road type
    road_type_speeds = {'FRC0': 0, 'FRC1': 0, 'FRC2': 0, 'FRC3': 0, 'FRC4': 0, 'FRC5': 0, 'FRC6': 0, 'FRC7': 0}

    # Compute the average speed for each road type
    for road_type, speeds in merged_speed_data.items():
        total_speed = 0
        count = 0
        for speed in speeds:
            time = datetime.datetime.strptime(speed['measurementTime'], '%Y-%m-%dT%H:%M:%SZ').time()
            if (time >= rush_start_time_am and time <= rush_end_time_am) or (time >= rush_start_time_pm and time <= rush_end_time_pm):
                # This is rush hour traffic
                total_speed += speed['speed']
                count += 1
        if count > 0:
            # There were measurements during rush hour
            average_speed = total_speed / count
            road_type_speeds[road_type] = average_speed

    # Load the road network from the GraphML file
    G = ox.load_graphml(graphml_file)

    # Add the average speeds to the edges in the road network
    for u, v, k, data in G.edges(keys=True, data=True):
        road_type = data['FRC']
        if road_type in road_type_speeds:
            data['speed'] = road_type_speeds.get(road_type, 0)

    return G
#===================================================================================================
#===================================================================================================
#===================================================================================================
#graph = loadGraph("HK", "travel_time", "drive", hwy_speeds)
#===================================================================================================
#===================================================================================================
#===================================================================================================

def shortestRoute(graph, orig:(float, float), dest:(float, float), optimizer="travel_time"):
    orig_node = ox.nearest_nodes(graph, orig[1], orig[0])
    dest_node = ox.nearest_nodes(graph, dest[1], dest[0])

    try:
        result = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
        #return nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)
    except Exception:
        #return nx.shortest_path(graph, dest_node, orig_node, weight=optimizer)
        
        #print("Entered in function shortestRoute - except")
        result = nx.shortest_path(graph, dest_node, orig_node, weight=optimizer)
        #print("orig and dest position exchanged")
    #result = nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)

    return result

def routeTimeForPrinting(graph, shortest_route, optimizer="travel_time"):
    route_time = int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, optimizer)))/60
    if route_time < 60:
        expression = str(int(route_time)) + " mins"
    elif route_time < 60*24:
        expression = str(int(route_time/60)) + " hours " + str(int(route_time) - int(route_time/60)*60) + " mins"
    return expression

def routeLength(graph, shortest_route):
    #return nx.shortest_path_length(G=graph, source=orig_node, target=dest_node, weight=weight)/1000
    return int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "length")))/1000

def routeTime(graph, shortest_route):
    return int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "travel_time")))/60
    #return int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, optimizer)))/60

def tunnelTest(nodes: int):
    if 582593392 in nodes or 587640136 in nodes:
        expression = "Cross Harbour Tunnel"
    else: expression = "None"
    return expression
# Tunnel Fees
# https://www.td.gov.hk/mini_site/atd/2020/en/section4_t_2.html

def getDepotFromDf(df):
    depot = df['origLoc'].mode()[0]
    return depot
#It should come from the only value from "origLoc"
#However, to avoid error, we add this function to secure the depot to be only one value

def tspSolver(df):
    return df
#this function would return the arrangement of a series of nodes

def getDistanceMatrix_backup(df, optimizer = "length"):

    distanceMatrix = []

    for orig in range(1, len(df)+1):
        distances = []
        for dest in range(1, len(df)+1):
            orig_lnglat = (float(df["destLoc"][orig].split(",")[0]), float(df["destLoc"][orig].split(",")[1]))
            dest_lnglat = (float(df["destLoc"][dest].split(",")[0]), float(df["destLoc"][dest].split(",")[1]))
            distance = routeLength_for_distMatrix(graph, orig_lnglat, dest_lnglat, "length")
            #print(orig_lnglat, "&", dest_lnglat, "@", distance)
            distances.append(distance)
        distanceMatrix.append(distances)

    return distanceMatrix

#check duplicate
def removeDupRoutes(routes, simRatio):
    paths = []
    for route in routes:
        path = list(route)
        if path not in paths: paths.append(list(route))

    for route_1 in range(len(paths)):
        for route_2 in range(route_1+1, len(paths)):
            similarity = difflib.SequenceMatcher(None, paths[route_1], paths[route_2])
            if (similarity.ratio() > simRatio) and (similarity.ratio()<1):
                paths[route_2] = []

    return [x for x in paths if x]

'''def loadGraph(place="HK", optimizer="travel_time", mode = "drive", hwy_speeds=hwy_speeds):
    ox.config(log_console = True, use_cache = True)
    #mode = 'drive' # 'drive', 'bike', 'walk'
    graph = ox.graph_from_place(place, network_type = mode)
    graph = ox.add_edge_speeds(graph, hwy_speeds)
    graph = ox.add_edge_travel_times(graph)
    return graph'''

def lngLatStrToFloat(point):
    lng = point.split(",")[0]
    lat = point.split(",")[1]
    return (float(lng), float(lat))

#-----------------------
#Ortools

#-----------------------

def randomColorGenerator(df):
    number_of_colors = len(df)
    colors = ["#"+''.join([random.choice('0123456789ABCD') for j in range(2)])+''.join([random.choice('0123456789') for j in range(2)])
        +''.join([random.choice('0123456789ABCDEF') for j in range(2)]) for i in range(number_of_colors)]
    return colors

def routeLength_for_distMatrix(graph, orig, dest, weight):
    orig_node = ox.nearest_nodes(graph, orig[1], orig[0])
    dest_node = ox.nearest_nodes(graph, dest[1], dest[0])
    return nx.shortest_path_length(G=graph, source=orig_node, target=dest_node, weight=weight)/1000

#global shortestRouteMap, shortestRouteLength, shortestRouteTime

#shortestRouteMap = shortestRouteLength = shortestRouteTime = 0

def VRP(graph, solution_list):
#in use    
    #define global variables
    #global shortestRouteMap, shortestRouteLength, shortestRouteTime
    shortest_RouteMap = shortest_RouteLength = shortest_RouteTime = 0

    colors = randomColorGenerator(solution_list)

    for node in range(len(solution_list)):
        route_length = route_time = 0
        if node == 0:
            #To skip the first node(depot)
            previous_node = node
            #continue
        #print(lngLatStrToFloat(solution_list[previous_node]), lngLatStrToFloat(solution_list[node]))
        
        if node == 1:
            #Initialise the map (Defining the map)
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[previous_node]), lngLatStrToFloat(solution_list[node]), "length")
            route_length = routeLength(graph, shortest_route)
            route_time = routeTime(graph, shortest_route)
            shortest_RouteMap = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap', color = colors[node])
            #popup_node = node
        elif solution_list[previous_node] != solution_list[node] and node!=0:
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[previous_node]), lngLatStrToFloat(solution_list[node]), "length")
            route_length = routeLength(graph, shortest_route)
            route_time = routeTime(graph, shortest_route) 
            shortest_RouteMap = ox.plot_route_folium(graph, shortest_route, route_map=shortest_RouteMap, tiles='openstreetmap', color = colors[node])
            if solution_list[node] == solution_list[0]:
                #If the node number is 0 means the truck is back to the depot
            #    popup_node = "Depot"
                marker = folium.Marker(location = lngLatStrToFloat(solution_list[node]), 
                                       tooltip=solution_list[node], popup="Depot") #latitude,longitude
                shortest_RouteMap.add_child(marker) 
            else: 
                #Main loop content
            #   popup_node = node
                marker = folium.Marker(location = lngLatStrToFloat(solution_list[node]), 
                                       tooltip=solution_list[node], popup=node) #latitude,longitude
                shortest_RouteMap.add_child(marker) 
        #marker = folium.Marker(location = lngLatStrToFloat(solution_list[node]), tooltip=solution_list[node], popup=popup_node) #latitude,longitude
        #shortest_RouteMap.add_child(marker) 
        #print(node, previous_node, lngLatStrToFloat(solution_list[previous_node]), lngLatStrToFloat(solution_list[node]), route_length, route_time)
        previous_node = node
        shortest_RouteLength += route_length
        shortest_RouteTime += route_time
    #print("VRP:", shortest_RouteLength, shortest_RouteTime)
    return shortest_RouteMap, shortest_RouteLength, shortest_RouteTime

def routeToLngLat(df, routinglist):
    for vehicle in range(len(routinglist)):
        for node in range(len(routinglist[vehicle])):
            if routinglist[vehicle][node] == 0: 
                routinglist[vehicle][node] = getDepotFromDf(df)
            else:
                routinglist[vehicle][node] = df["destLoc"][routinglist[vehicle][node]]
    return routinglist

def decode_route(list_lnglat: list, list_order: list):
    route_order = []
    first_element = False
    for order in list_order:
        if first_element:
            distance = ((float(route_order[-1].split(",")[0]) - float(list_lnglat[order].split(",")[0]))**2 
                        + (float(route_order[-1].split(",")[1]) - float(list_lnglat[order].split(",")[1]))**2)**(1/2)*110.948/10
            #print("distance:", distance)
            #below 100m then skip
            #if list_lnglat[order] != route_order[-1]: route_order.append(list_lnglat[order])
            if distance >= 0.1: route_order.append(list_lnglat[order])
        else:
            route_order.append(list_lnglat[order])
            first_element = True
    #print("route_order", route_order)
    return route_order

#['22.45055, 114.01209', '22.45055, 114.01209', '22.44591, 114.03469']
#0120
#oln list: ['22.45055, 114.01209', '22.45055, 114.01209', '22.44591, 114.03469', '22.45055, 114.01209']


# In[14]:


#graph = loadGraph("HK", "travel_time", "drive", hwy_speeds)
graph = loadGraph("hongkong_speed.graphml")


# In[15]:


def getDistanceMatrix(routelist, optimizer = "length"):

    #nodes = result = list(getDepotFromDf(df).split("-"))
    #for i in df["destLoc"]:
    #    nodes.append(i)
    routeList = routelist[:len(routelist)]
    distanceMatrix = []

    for orig in routeList:
        distances = []
        for dest in routeList:
            if orig == dest: 
                distances.append(0)
                continue
            else:
                orig_lnglat = (float(orig.split(",")[0]), float(orig.split(",")[1]))
                dest_lnglat = (float(dest.split(",")[0]), float(dest.split(",")[1]))
                shortest_route = shortestRoute(graph, orig_lnglat, dest_lnglat)
                #distance = routeLength_for_distMatrix(graph, orig_lnglat, dest_lnglat, "length")
                distance = routeLength(graph, shortest_route)
                shortest_route = 0
            #print(orig_lnglat, "&", dest_lnglat, "@", distance)
            distances.append(distance)
        distanceMatrix.append(distances)
    return distanceMatrix

#ortool testing
"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

def create_data_model(list):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = getDistanceMatrix(list, "length")
    
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    #print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    routingSol = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        #plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0

        routeSol = []
        while not routing.IsEnd(index):
            #plan_output += ' {} -> '.format(manager.IndexToNode(index))
            routeSol.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        #plan_output += '{}\n'.format(manager.IndexToNode(index))
        routeSol.append(manager.IndexToNode(index))
        #plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        #print(plan_output)
        routingSol.append(routeSol)
        #routingSol.append(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    #print('Maximum of the route distances: {}m'.format(max_route_distance))
    return routingSol[0]

def get_routes(solution, routing, manager):
  """Get vehicle routes from a solution and store them in an array."""
  # Get vehicle routes and store them in a two dimensional array whose
  # i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route)
  return routes

def ortool(list):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(list)
    result = []

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        300000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        result = print_solution(data, manager, routing, solution)
    else:
        print('No solution found !')
    
    return result


#if __name__ == '__main__':
    #resultingSol = main()


# In[16]:


raw_clean.head()


# In[17]:


#clear the route that contains the point outside hong kong
#str = '22.1504, 113.55207'

from shapely.geometry import Point, Polygon

# Define Hong Kong's boundary polygon using its latitude and longitude coordinates
hong_kong_boundary = Polygon([(22.1535, 113.8259), (22.1535, 114.4214), (22.5609, 114.4214),
                              (22.5176, 114.2667), (22.5448, 114.1629), (22.5008, 114.0857),
                              (22.5223, 113.9295), (22.1535, 113.8259)])

invalid_routes = []

for idx, row in raw_clean.iterrows():
    points = row['Route']
    inside = True
    for point in points:
        is_inside_hong_kong = hong_kong_boundary.contains(Point(float(point.split(",")[0]), float(point.split(",")[1])))
        if is_inside_hong_kong == False:
            inside = False
            break
    if inside == False:
        invalid_routes.append(idx)

raw_clean = raw_clean[~raw_clean.index.isin(invalid_routes)]
raw_clean.shape


# In[18]:


from tqdm import tqdm

distances = []
times = []
maps = []
journey_orders = []
#print("length=", len(raw_clean))

for idx, row in tqdm(raw_clean.iterrows(), total=len(raw_clean)):
    journey = row["Route"]
    route_order = ortool(journey)
    journey_order = decode_route(journey, route_order)
    shortestRouteMap, shortestRouteLength, shortestRouteTime = VRP(graph, journey_order)
    journey_orders.append(journey_order)
    distances.append(shortestRouteLength)
    times.append(shortestRouteTime)
    maps.append(shortestRouteMap)
    #print("idx:", idx, shortestRouteLength, shortestRouteTime)

df_journey_orders = pd.DataFrame({"Route_Order":journey_orders})
df_distances = pd.DataFrame({"Distances":distances})
df_times = pd.DataFrame({"Time":times})
df_maps = pd.DataFrame({"Maps":maps})

raw_clean = raw_clean.reset_index()
raw_clean = raw_clean.assign(Route_Order=df_journey_orders["Route_Order"],
                             Distances=df_distances["Distances"],
                             Time=df_times["Time"],
                             Maps=df_maps["Maps"])
raw_clean.head()


# In[ ]:


raw_clean.to_csv("raw_clean_osm_hk_all.csv")


# In[ ]:


#weather handling
#download from https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

import pandas as pd
import numpy as np
from shapely.geometry import Point

# Set Pandas options to show all columns
pd.set_option('display.max_columns', None)

# Read the weather data file into a dataframe
df_weather_hk = pd.read_csv("45007_2016_2023.csv")

# Select the desired columns and merge the latitude and longitude columns into a point
df_weather_hk = df_weather_hk[["STATION", "DATE", "NAME", "LATITUDE", "LONGITUDE", "TEMP", "DEWP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP", "FRSHTT"]]
df_weather_hk["POINT"] = df_weather_hk.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)

# Replace 999.9 values in GUST column with 0
df_weather_hk.loc[df_weather_hk["GUST"] == 999.9, "GUST"] = 0

# Replace 99.9 values in PRCP column with 0
df_weather_hk.loc[df_weather_hk["PRCP"] == 99.9, "PRCP"] = 0

# Replace 999.9 values in SNDP column with 0
df_weather_hk.loc[df_weather_hk["SNDP"] == 999.9, "SNDP"] = 0

# Convert FRSHTT column to string and fill 0 at the front if necessary
df_weather_hk["FRSHTT"] = df_weather_hk["FRSHTT"].astype(str).str.zfill(6)

# Split FRSHTT column into 6 separate columns with proper naming
df_weather_hk = pd.concat([df_weather_hk, df_weather_hk["FRSHTT"].apply(lambda x: pd.Series(list(x)))], axis=1)
df_weather_hk = df_weather_hk.rename(columns={0: "FOG_MIST", 1: "PRECIP_DRIZZLE", 2: "HAIL_SLEET", 3: "THUNDER", 4: "TORNADO_FUNNEL_CLOUD", 5: "WIND_DAMAGE"})

# Drop the original FRSHTT column and the LATITUDE and LONGITUDE columns
df_weather_hk = df_weather_hk.drop(columns=["FRSHTT", "LATITUDE", "LONGITUDE"])

# Modify the NAME column as requested
df_weather_hk["NAME"] = df_weather_hk["NAME"].apply(lambda x: "HONG KONG" if x == "HONG KONG INTERNATIONAL, HK" else x)

# Drop any rows where the NAME column is not equal to "HONG KONG INTERNATIONAL, HK"
df_weather_hk = df_weather_hk[df_weather_hk["NAME"] == "HONG KONG"]

# Convert STATION column to string and keep only the first five digits
df_weather_hk["STATION"] = df_weather_hk["STATION"].astype(str).str[:5]

# Move the POINT column to be right after the STATION column
cols = list(df_weather_hk.columns)
cols.remove("POINT")
cols.insert(cols.index("STATION")+1, "POINT")
df_weather_hk = df_weather_hk.reindex(columns=cols)

# Reorder the columns to have DATE, NAME, STATION, POINT, and then the remaining columns
df_weather_hk = df_weather_hk[["DATE", "NAME", "STATION", "POINT", "TEMP", "DEWP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP", "FOG_MIST", "PRECIP_DRIZZLE", "HAIL_SLEET", "THUNDER", "TORNADO_FUNNEL_CLOUD", "WIND_DAMAGE"]]

# Print the resulting dataframe
df_weather_hk.head()


# In[ ]:


# Convert DATE column in df_weather_hk to datetime format
df_weather_hk['DATE'] = pd.to_datetime(df_weather_hk['DATE'])

# Convert planStartTime column in raw_clean to datetime format that matches DATE format
raw_clean['planStartTime'] = pd.to_datetime(raw_clean['planStartTime'].dt.strftime('%Y-%m-%d'))

# Merge raw_clean and df_weather_hk based on planStartTime and DATE columns using merge_asof()
raw_merge = pd.merge_asof(raw_clean.sort_values('planStartTime'), df_weather_hk, left_on='planStartTime', right_on='DATE', direction='nearest')

# Drop the unused columns
raw_merge = raw_merge.drop(columns=['DATE', 'NAME', 'STATION', 'POINT'])

# Print the resulting dataframe
raw_merge.head()


# In[ ]:


raw_merge.to_csv("raw_merge_osm_weather_hk_all.csv")


# In[ ]:




