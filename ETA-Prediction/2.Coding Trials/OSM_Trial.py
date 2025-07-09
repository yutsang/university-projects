#!/usr/bin/env python
# coding: utf-8

# In[9]:


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

#from folium.features import DivIcon

pd.set_option('display.max_rows', None)

df_backup = pd.read_csv("C:\Users\tsang\Desktop\Github\ETA-Prediction-Data\Data\eta_prediction.csv", encoding='UTF-8')
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
df["planTaskTime"] = df["Time-5"].dropna()-df["Time-1"].dropna()
df["actTaskTime"] = df["Time-5"].dropna()-df["Time-3"].dropna()
#df["plannedRemainTime"] = df["Time-5"].dropna().astype(datetime64)-df["Time-2"].dropna().astype(datetime64)
#df["actualRemainTime"] = df["Time-5"].dropna().astype(datetime64)-df["Time-4"].dropna().astype(datetime64)

df = df.rename({"Time-1": "planStartTime", "Time-2": "planEndTime", "Time-3": "actStartTime", 
    "Time-4": "actEndTime", "Time-5": "MiscTime"}, axis = "columns")

df["origLoc"] = df[["Start-Lng", "Start-Lat"]].agg(', '.join, axis=1)
df["destLoc"] = df[["Dest-Lng", "Dest-Lat"]].agg(', '.join, axis=1)
df = df.drop(["Start-Lng", "Start-Lat", "Dest-Lng", "Dest-Lat"], axis=1)

cols = ["Loc", "Vehicle", "ShipNo", "origLoc", "destLoc", "planStartTime", "planEndTime", "actStartTime", "actEndTime", 
    "MiscTime", "planDurn", "actDurn", "planTaskTime", "actTaskTime"]

df = df[cols]


# In[1]:


#python -m pip install ortools
#pip install folium
#pip install osmnx


# In[ ]:


#Initialise -> will put to somewhere later
hwy_speeds = {"residential": 10, "secondary": 33, "tertiary": 60}

def loadGraph(place="HK", optimizer="travel_time", mode = "drive", hwy_speeds=hwy_speeds):
    ox.config(log_console = True, use_cache = True)
    #mode = 'drive' # 'drive', 'bike', 'walk'
    graph = ox.graph_from_place(place, network_type = mode)
    graph = ox.add_edge_speeds(graph, hwy_speeds)
    graph = ox.add_edge_travel_times(graph)
    return graph

def shortestRoute(graph, orig:(float, float), dest:(float, float), optimizer="travel_time"):
    orig_node = ox.nearest_nodes(graph, orig[1], orig[0])
    dest_node = ox.nearest_nodes(graph, dest[1], dest[0])
    return nx.shortest_path(graph, orig_node, dest_node, weight=optimizer)

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
            distance = routeLength(graph, orig_lnglat, dest_lnglat, "length")
            #print(orig_lnglat, "&", dest_lnglat, "@", distance)
            distances.append(distance)
        distanceMatrix.append(distances)

    return distanceMatrix

def getDistanceMatrix(df, optimizer = "length"):

    nodes = result = list(getDepotFromDf(df).split("-"))
    for i in df["destLoc"]:
        nodes.append(i)

    distanceMatrix = []

    for orig in range(len(nodes)):
        distances = []
        for dest in range(len(nodes)):
            orig_lnglat = (float(nodes[orig].split(",")[0]), float(nodes[orig].split(",")[1]))
            dest_lnglat = (float(nodes[dest].split(",")[0]), float(nodes[dest].split(",")[1]))
            distance = routeLength_for_distMatix(graph, orig_lnglat, dest_lnglat, "length")
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

def loadGraph(place="HK", optimizer="travel_time", mode = "drive", hwy_speeds=hwy_speeds):
    ox.config(log_console = True, use_cache = True)
    #mode = 'drive' # 'drive', 'bike', 'walk'
    graph = ox.graph_from_place(place, network_type = mode)
    graph = ox.add_edge_speeds(graph, hwy_speeds)
    graph = ox.add_edge_travel_times(graph)
    return graph

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

#Not in use
def VRP_v1(graph, solution_list):
    
    shortestRouteLength = shortestRouteTime = 0

    colors = randomColorGenerator(solution_list[0])

    for node in range(len(solution_list[0])):
        if node == 0:
            #To skip the first node(depot)
            previous_node = node
            continue
        if node == 1:
            #Initialise the map (Defining the map)
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[0][previous_node]), lngLatStrToFloat(solution_list[0][node]), "length")
            shortestRouteMap = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap', color = colors[node]) 
            marker = folium.Marker(location = lngLatStrToFloat(solution_list[0][node]), tooltip=solution_list[0][node], popup=node) #latitude,longitude
            shortestRouteMap.add_child(marker) 
            marker = folium.Marker(location = lngLatStrToFloat(solution_list[0][previous_node]), tooltip=solution_list[0][previous_node], popup="Depot")
        if solution_list[0][node] == solution_list[0][0]:
            #If the node number is 0 means the truck is back to the depot
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[0][previous_node]), lngLatStrToFloat(solution_list[0][node]), "length")
            shortestRouteMap = ox.plot_route_folium(graph, shortest_route, route_map=shortestRouteMap, tiles='openstreetmap', color = colors[node])
            marker = folium.Marker(location = lngLatStrToFloat(solution_list[0][node]), tooltip=solution_list[0][node], popup="Depot") #latitude,longitude
            #marker = folium.Marker(location = lngLatStrToFloat(resultingSolTest[0][node]), tooltip=resultingSolTest[0][node], icon=DivIcon(
            #icon_size=(150,36), icon_anchor=(7,20), html='<div style="font-size: 18pt; color : black">'+str(node)+'</div>',)) #Number Markers
            shortestRouteMap.add_child(marker) 
        elif solution_list[0][previous_node] != solution_list[0][node]:
            #Main loop content
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[0][previous_node]), lngLatStrToFloat(solution_list[0][node]), "length")
            shortestRouteMap = ox.plot_route_folium(graph, shortest_route, route_map=shortestRouteMap, tiles='openstreetmap', color = colors[node], opacity=0.5)
            marker = folium.Marker(location = lngLatStrToFloat(solution_list[0][node]), tooltip=solution_list[0][node], popup=node) #latitude,longitude
            #marker = folium.Marker(location = lngLatStrToFloat(resultingSolTest[0][node]), tooltip=resultingSolTest[0][node], icon=DivIcon(
            #icon_size=(150,36), icon_anchor=(7,20), html='<div style="font-size: 18pt; color : black">'+str(node)+'</div>',)) #Number Markers
            shortestRouteMap.add_child(marker) 
        previous_node = node
    return shortestRouteMap, shortestRouteLength, shortestRouteTime


def routeLength_for_distMatix(graph, orig, dest, weight):
    orig_node = ox.nearest_nodes(graph, orig[1], orig[0])
    dest_node = ox.nearest_nodes(graph, dest[1], dest[0])
    return nx.shortest_path_length(G=graph, source=orig_node, target=dest_node, weight=weight)/1000
'''
def routeTime(graph, shortest_route, optimizer="travel_time"):
    return int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, optimizer)))/60


'''

def VRP(graph, solution_list):
    
    shortestRouteLength = shortestRouteTime = 0

    colors = randomColorGenerator(solution_list[0])

    for node in range(len(solution_list[0])):
        route_length = route_time = 0
        if node == 0:
            #To skip the first node(depot)
            previous_node = node
            continue
        shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[0][previous_node]), lngLatStrToFloat(solution_list[0][node]), "length")
        route_length = routeLength(graph, shortest_route)
        route_time = routeTime(graph, shortest_route)
        if node == 1:
            #Initialise the map (Defining the map)
            shortestRouteMap = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap', color = colors[node])
            popup = node
        elif solution_list[0][previous_node] != solution_list[0][node]: 
            shortestRouteMap = ox.plot_route_folium(graph, shortest_route, route_map=shortestRouteMap, tiles='openstreetmap', color = colors[node])
            if solution_list[0][node] == solution_list[0][0]:
                #If the node number is 0 means the truck is back to the depot
                popup = "Depot"
            else: 
                #Main loop content
                popup = node
        marker = folium.Marker(location = lngLatStrToFloat(solution_list[0][node]), tooltip=solution_list[0][node], popup=popup) #latitude,longitude
        shortestRouteMap.add_child(marker) 
        print(node, previous_node, lngLatStrToFloat(solution_list[0][previous_node]), lngLatStrToFloat(solution_list[0][node]), route_length, route_time)
        previous_node = node
        shortestRouteLength += route_length
        shortestRouteTime += route_time
    return shortestRouteMap, shortestRouteLength, shortestRouteTime

def routeToLngLat(df, routinglist):
    for vehicle in range(len(routinglist)):
        for node in range(len(routinglist[vehicle])):
            if routinglist[vehicle][node] == 0: 
                routinglist[vehicle][node] = getDepotFromDf(df)
            else:
                routinglist[vehicle][node] = df["destLoc"][routinglist[vehicle][node]]
    return routinglist


# In[ ]:


#Coding

#initialization only
graph = loadGraph("HK", "travel_time", "drive", hwy_speeds)
orig = (22.36991, 114.13556)
dest = (22.28042, 114.18443)
shortest_route = shortestRoute(graph, orig, dest, )

#min
#route_time = routeTime(graph, shortest_route, "travel_time")
route_time = routeTime(graph, shortest_route)
#km
#length = routeLength(graph, orig, dest, 'length')
length = routeLength(graph, shortest_route)

shortest_route_map = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap')


print(length)
print(route_time)
shortest_route_map


# In[ ]:


#Current Strategy for Testing the Data
#data set = HKG, Date = 2019-01-02, Time = 11:40:55

df_HKG = df[df["Loc"] == "HKG"]
df_HKG["Date"] = df_HKG["planStartTime"].dt.date.astype(str)
df_HKG = df_HKG[df_HKG["Date"] == "2019-01-02"].drop({"Date"}, axis=1)
df_HKG["Time"] = df_HKG["planStartTime"].dt.time.astype(str)
df_HKG = df_HKG[df_HKG["Time"] == "11:40:55"].drop({"Time"}, axis=1).sort_values(by=["planStartTime"])
df_HKG = df_HKG.drop_duplicates(subset=['destLoc'], keep="first", inplace=False)
df_HKG.index = np.arange(1, len(df_HKG) + 1)

#Assume only go to a dest once per journey
df_HKG
#1.Actual Start Time >= Planned Start Time
#2. Plan Duration > 0
#3. Actual Duration > 0
#4. Time-4 Not Null
#5. Check Time 1-3 Null? If null then drop
#6. Cal journey time by OSM, and set threshold for the actual Duration, intial 10 times


# In[ ]:


#ortool testing
"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = getDistanceMatrix(df_HKG, "length")
    
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    routingSol = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0

        routeSol = []
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            routeSol.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        routeSol.append(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        routingSol.append(routeSol)
        #routingSol.append(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))
    return routingSol

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



def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()
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


if __name__ == '__main__':
    resultingSol = main()


# In[ ]:


#check duplicate routes

shortest_route_4 = nx.all_shortest_paths(graph, ox.nearest_nodes(graph, 114.180560, 22.320670), 
    ox.nearest_nodes(graph, 114.146118, 22.284639), weight=None, method="travel_time")
graph = ox.add_edge_speeds(graph, hwy_speeds)
graph = ox.add_edge_travel_times(graph)

colors = ["red", "blue", "green"]


shortest_route_4 = removeDupRoutes(shortest_route_4, 0.9)
i = 0
for path in shortest_route_4:
    print(path)
    if i == 0: 
        shortest_route_map_1 = ox.plot_route_folium(graph, path, tiles='openstreetmap', color = colors[i])
    else: 
        shortest_route_map_1 = ox.plot_route_folium(graph, path, route_map= shortest_route_map_1, tiles='openstreetmap', color = colors[i])
    i += 1

print(i)
shortest_route_map_1


# In[ ]:


'''print(resultingSol)
print(routeToLngLat(df_HKG, resultingSol))'''


# In[ ]:


hwy_speeds = {"residential": 10, "secondary": 33, "tertiary": 60}
graph = loadGraph("HK", "travel_time", "drive", hwy_speeds)

shortestRouteMap, shortestRouteLength, shortestRouteTime = VRP(graph, resultingSol)
print(shortestRouteLength, "km")
print(shortestRouteTime, "mins")
shortestRouteMap


# In[ ]:


#config

hwy_speeds = {"residential": 10, "secondary": 33, "tertiary": 60}
graph = loadGraph("HK", "travel_time", "drive", hwy_speeds)

optimizer = ["travel_time", "time", "length"]
i = 0

point_A = [[22.45055, 114.01209], [22.39502, 113.97302]]
shortest_route = shortestRoute(graph, point_A[0], point_A[1], optimizer[i])
print(round(routeLength(graph, shortest_route), 1), "(#9.4km)", ";", round(routeTime(graph, shortest_route), 1), "#12mins")

point_B = [[22.39502, 113.97302], [22.20682, 114.02839]]
shortest_route = shortestRoute(graph, point_B[0], point_B[1], optimizer[i])
print(round(routeLength(graph, shortest_route), 1), "#32.8km", ";", round(routeTime(graph, shortest_route), 1), "#43mins")

point_C = [[22.20682, 114.02839], [22.37382, 114.11763]]
shortest_route = shortestRoute(graph, point_C[0], point_C[1], optimizer[i])
print(round(routeLength(graph, shortest_route), 1), "#37.8km", ";", round(routeTime(graph, shortest_route), 1), "#44mins")

point_D = [[22.37382, 114.11763], [22.35673, 114.12773]]
shortest_route = shortestRoute(graph, point_D[0], point_D[1], optimizer[i])
print(round(routeLength(graph, shortest_route), 1), "#3.6km", ";", round(routeTime(graph, shortest_route), 1), "#5mins")


# In[ ]:


distances = getDistanceMatrix(df_HKG, optimizer = "length")
distances

