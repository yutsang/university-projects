import folium
import osmnx as ox
import networkx as nx

def plot_shipment_route(row, region_graphml_path):
    """
    Given a shipment row and a path to a region's OSM graphml file, return a folium map with the route.
    """
    # Parse origin and destination
    orig = tuple(map(float, row['origLoc'].split(',')))
    dest = tuple(map(float, row['destLoc'].split(',')))
    # Load graph
    graph = ox.load_graphml(region_graphml_path)
    # Find nearest nodes
    orig_node = ox.nearest_nodes(graph, orig[0], orig[1])
    dest_node = ox.nearest_nodes(graph, dest[0], dest[1])
    # Get shortest path
    route = nx.shortest_path(graph, orig_node, dest_node, weight='travel_time')
    # Create map
    m = folium.Map(location=orig, zoom_start=12)
    # Plot route
    folium.PolyLine(locations=[(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in route], color='blue').add_to(m)
    folium.Marker(location=orig, popup='Origin', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=dest, popup='Destination', icon=folium.Icon(color='red')).add_to(m)
    return m 