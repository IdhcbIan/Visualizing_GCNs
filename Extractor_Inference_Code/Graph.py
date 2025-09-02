
import json
import os
import numpy as np


def plot_and_export(rankings, k_list, model_name):
    for k in k_list:

        G, pos =  visualize_knn_graph(rankings, k=k) 
        export_graph_positions(G, pos, model_name, k)


def visualize_knn_graph(rankings, k=16, figsize=(15, 10), title=None, 
                        show_labels=False, save_path=None):
    """
    Visualize a KNN graph based on rankings using spring layout.
    
    Parameters:
    -----------
    rankings : numpy.ndarray
        The ranking indices from BallTree or similar
    k : int
        Number of neighbors to visualize for each node
    figsize : tuple
        Size of the figure
    title : str
        Title of the plot
    node_colors : list
        Colors for nodes (default colors are based on classification)
    show_labels : bool
        Whether to show node labels
    save_path : str
        Path to save the figure
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(len(rankings)):
        G.add_node(i)
    
    # Add edges based on k-nearest neighbors
    for i in range(len(rankings)):
        # Skip the first neighbor as it's the point itself
        for j in range(1, min(k+1, len(rankings[i]))):
            if i != rankings[i][j]: 
                G.add_edge(i, rankings[i][j])
    
    # Create spring layout with increased spacing between nodes
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    #print(pos)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes with colors
    node_sizes = [100 if i < 200 else 150 for i in range(len(rankings))]  # Larger nodes for test data
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=False)
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'KNN Graph (k={k})')
    
    plt.axis('off')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print some graph statistics
    print(f"Graph info: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G, pos



def export_graph_positions(G, pos,model_name, k):
    """
    Export the graph node positions to a JSON file.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph whose positions will be exported
    pos : dict
        Dictionary of node positions {node_id: (x, y)}
    output_path : str
        Path where the JSON file will be saved
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    # Create a serializable dictionary of positions
    # Convert numpy arrays to lists for JSON serialization
    export_data = {
        "nodes": {},
        "graph_info": {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges)
        }
    }
    
    # Add node positions and any attributes
    for node_id in G.nodes():
        node_pos = pos[node_id]
        # Convert numpy arrays or tuples to lists for JSON serialization
        position = [float(node_pos[0]), float(node_pos[1])]
        
        # Include node attributes if available
        node_attrs = dict(G.nodes[node_id])
        export_data["nodes"][str(node_id)] = {
            "position": position,
            "attributes": node_attrs
        }
    
   
    path = model_name + "_output_"+ str(k) + ".json"
    print("Path: ")
    print(path)
    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Graph positions exported to {path} with k: {k}")
    return path


# Export the positions
#output_path = "Runs/graph_positions.json"
#export_graph_positions(G, pos, output_path)
# 
