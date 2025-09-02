def visualize_knn_graph(rankings, k=16, figsize=(15, 10), title=None, node_colors=None, 
                        show_labels=False, layout='spring', save_path=None):
    """
    Visualize a KNN graph based on rankings.
    
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
    layout : str
        Graph layout algorithm ('spring', 'kamada_kawai', 'circular', etc.)
    save_path : str
        Path to save the figure
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    
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
    
    # Setup node colors if not provided
    if node_colors is None:
        colors = []
        for i in range(len(rankings)):
            if 0 <= i < 100:  # Yes train
                colors.append('red')
            elif 100 <= i < 200:  # No train
                colors.append('green')
            elif 200 <= i < 210:  # Yes test
                colors.append('purple')
            elif 210 <= i < 220:  # No test
                colors.append('cyan')
            else:  # Handle any other indices
                colors.append('gray')
    else:
        colors = node_colors
    
    # Layout options
    layout_funcs = {
        'spring': nx.spring_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'circular': nx.circular_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout
    }
    
    # Create layout
    layout_func = layout_funcs.get(layout, nx.spring_layout)
    pos = layout_func(G)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes with colors
    node_sizes = [100 if i < 200 else 150 for i in range(len(rankings))]  # Larger nodes for test data
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=False)
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Train - Blocked'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Train - Not Blocked'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Test - Blocked'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='Test - Not Blocked')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'KNN Graph (k={k})')
    
    plt.axis('off')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Use plt.savefig instead of plt.show() for non-interactive environments
    plt.savefig('knn_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some graph statistics
    print(f"Graph info: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

# Non-interactive function to visualize with different k values
def batch_knn_visualization(rankings, k_values=[5, 10, 15], layouts=['spring']):
    """
    Generates multiple visualizations with different k values and layouts
    """
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "knn_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations for each combination
    for k in k_values:
        for layout in layouts:
            save_path = os.path.join(output_dir, f"knn_graph_k{k}_{layout}.png")
            print(f"Generating visualization with k={k}, layout={layout}")
            visualize_knn_graph(
                rankings, 
                k=k, 
                layout=layout, 
                show_labels=(k < 10),  # Only show labels for small k
                title=f'KNN Graph (k={k}, layout={layout})',
                save_path=save_path
            )
    
    print(f"All visualizations saved to {output_dir}/")

# Example usage
# Load the rankings from JSON
import json
import os
import numpy as np

try:
    with open("./Runs/resnet152_output.json", "r") as f:
        rankings = json.load(f)
    
    # Convert to numpy array if needed
    rankings = np.array(rankings)
    
    # Call the batch visualization function with different parameters
    batch_knn_visualization(
        rankings, 
        k_values=[5, 10, 20], 
        layouts=['spring', 'circular']
    )
    
    print("Visualization complete. Check the knn_visualizations directory for output images.")
    
except FileNotFoundError:
    print("Rankings file not found. Please ensure resnet152_output.json exists in the Runs directory.")