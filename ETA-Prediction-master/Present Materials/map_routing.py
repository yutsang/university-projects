import pandas as pd
import warnings
import json
import datetime
import osmnx as ox
import random
import folium
import networkx as nx

warnings.filterwarnings('ignore')

df = pd.read_csv("intermediate_v2.csv", index_col = 0)

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

def lngLatStrToFloat(point):
    lng = point[0]
    lat = point[1]
    return (float(lng), float(lat))


def VRP(graph, solution_list):
#in use    
    #define global variables
    #global shortestRouteMap, shortestRouteLength, shortestRouteTime
    shortest_RouteMap = shortest_RouteLength = shortest_RouteTime = 0

    number_of_colors = len(df)
    colors = ["#"+''.join([random.choice('0123456789ABCD') for j in range(2)])+''.join([random.choice('0123456789') for j in range(2)])
        +''.join([random.choice('0123456789ABCDEF') for j in range(2)]) for i in range(number_of_colors)]
    
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
            route_length = int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "length")))/1000
            route_time = int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "travel_time")))/60
            shortest_RouteMap = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap', color = colors[node])
            #popup_node = node
        elif solution_list[previous_node] != solution_list[node] and node!=0:
            shortest_route = shortestRoute(graph, lngLatStrToFloat(solution_list[previous_node]), lngLatStrToFloat(solution_list[node]), "length")
            route_length = int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "length")))/1000
            route_time = int(sum(ox.utils_graph.get_route_edge_attributes(graph, shortest_route, "travel_time")))/60 
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

#change the routing format to coordinates
def decode_route_list(lat_lng_str):
    # Convert the string to a list of strings
    lat_lng_list = eval(lat_lng_str)
    # Convert the list of strings to a list of tuples containing floats
    lat_lng_tuples = [(float(lat), float(lng)) for lat, lng in (pair.split(',') for pair in lat_lng_list)]
    return lat_lng_tuples

# graph = loadGraph("hongkong_speed.graphml")

# shortestRouteMap, shortestRouteLength, shortestRouteTime = VRP(graph, decode_route_list(df.iloc[0]["Route_Order"]))
# #input graph, list of ordered route
# print(shortestRouteMap)