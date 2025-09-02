import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm

def load_knn_data(file_path):
    """Load KNN rankings from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def build_graph(knn_data, k=5):
    """Build a NetworkX graph from KNN data."""
    G = nx.Graph()
    
    # Add nodes
    for idx, item in enumerate(knn_data):
        original_id = item[0]
        G.add_node(idx, original_id=original_id, name=f"Node {idx}")
    
    # Add edges based on KNN rankings
    for source_idx, rankings in enumerate(knn_data):
        if not isinstance(rankings, list) or len(rankings) <= 1:
            continue
            
        # Connect to k nearest neighbors (or fewer if not enough rankings)
        for i in range(1, min(k + 1, len(rankings))):
            target_original_id = rankings[i]
            target_idx = None
            
            # Find the target node index in our dataset
            for node_idx, node_data in G.nodes(data=True):
                if node_data['original_id'] == target_original_id:
                    target_idx = node_idx
                    break
            
            if target_idx is not None:
                # Add edge with weight based on rank (higher weight = closer relationship)
                weight = 1.0 / (i + 1)
                G.add_edge(source_idx, target_idx, weight=weight)
    
    return G

def find_connected_components(G):
    """Find all connected components (islands) in the graph."""
    return list(nx.connected_components(G))

def visualize_graph(G, k, output_dir, export_positions=True, figsize=(12, 10)):
    """Visualize the graph with force-directed layout and highlight islands."""
    plt.figure(figsize=figsize)
    
    # Find connected components (islands)
    islands = list(nx.connected_components(G))
    print(f"Found {len(islands)} islands in the graph")
    
    # Generate colors for islands
    colors = plt.cm.rainbow(np.linspace(0, 1, len(islands)))
    
    # Create node color map
    node_colors = []
    for node in G.nodes():
        for i, island in enumerate(islands):
            if node in island:
                node_colors.append(colors[i])
                break
    
    # Use force-directed layout with stronger clustering
    # Adjust the parameters to get different levels of clustering
    pos = nx.spring_layout(
        G, 
        k=0.5,         # Optimal distance between nodes (smaller = tighter clusters)
        iterations=100, # More iterations for better layout
        seed=42        # For reproducibility
    )
    
    # Draw edges with transparency based on weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    
    # Only show labels for a subset of nodes to avoid overcrowding
    # Alternatively, you can enable this for all nodes but might be cluttered
    # max_labels = min(30, len(G.nodes()))
    # node_subset = list(G.nodes())[:max_labels]
    # node_labels = {node: G.nodes[node]['name'] for node in node_subset}
    # nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title(f"KNN Graph Visualization (k={k})")
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'knn_graph_k{k}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export node positions if requested
    if export_positions:
        positions = {}
        for node, coords in pos.items():
            original_id = G.nodes[node]['original_id']
            positions[original_id] = {
                'x': float(coords[0]),
                'y': float(coords[1]),
                'node_id': node,
                'island': None
            }
            
            # Add island information
            for i, island in enumerate(islands):
                if node in island:
                    positions[original_id]['island'] = i
                    break
        
        with open(os.path.join(output_dir, f'node_positions_k{k}.json'), 'w') as f:
            json.dump(positions, f, indent=2)
            
        print(f"Node positions exported to {os.path.join(output_dir, f'node_positions_k{k}.json')}")
    
    return positions, islands

def run_tsne_embedding(knn_data, k, perplexity=30, output_dir='output'):
    """Use t-SNE to create a 2D embedding based on the KNN relationships."""
    print("Creating t-SNE embedding from KNN data...")
    
    # Create a sparse similarity matrix from KNN rankings
    n = len(knn_data)
    similarity_matrix = np.zeros((n, n))
    
    # Fill the similarity matrix based on KNN rankings
    for i, rankings in enumerate(tqdm(knn_data, desc="Building similarity matrix")):
        if not isinstance(rankings, list) or len(rankings) <= 1:
            continue
            
        for j in range(1, min(k + 1, len(rankings))):
            target_id = rankings[j]
            # Find target index
            target_idx = None
            for idx, item in enumerate(knn_data):
                if item[0] == target_id:
                    target_idx = idx
                    break
                    
            if target_idx is not None:
                # Higher ranking = higher similarity
                similarity = 1.0 / (j + 1)
                similarity_matrix[i, target_idx] = similarity
                similarity_matrix[target_idx, i] = similarity  # Make it symmetric
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(similarity_matrix)
    
    # Create a graph from the embedding
    G = nx.Graph()
    
    # Add nodes with positions from t-SNE
    for i, coords in enumerate(embedded):
        original_id = knn_data[i][0]
        G.add_node(i, original_id=original_id, name=f"Node {i}", pos=coords)
    
    # Add edges based on KNN rankings
    for source_idx, rankings in enumerate(knn_data):
        if not isinstance(rankings, list) or len(rankings) <= 1:
            continue
            
        for j in range(1, min(k + 1, len(rankings))):
            target_id = rankings[j]
            # Find target index
            target_idx = None
            for idx, item in enumerate(knn_data):
                if item[0] == target_id:
                    target_idx = idx
                    break
                    
            if target_idx is not None:
                weight = 1.0 / (j + 1)
                G.add_edge(source_idx, target_idx, weight=weight)
    
    # Get node positions from the embedding
    pos = {i: (coords[0], coords[1]) for i, coords in enumerate(embedded)}
    
    # Find connected components (islands)
    islands = list(nx.connected_components(G))
    print(f"Found {len(islands)} islands in the t-SNE embedding")
    
    # Generate colors for islands
    colors = plt.cm.rainbow(np.linspace(0, 1, len(islands)))
    
    # Create node color map
    node_colors = []
    for node in G.nodes():
        for i, island in enumerate(islands):
            if node in island:
                node_colors.append(colors[i])
                break
    
    # Plot the embedding with islands highlighted by color
    plt.figure(figsize=(12, 10))
    
    # Draw edges with transparency based on weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    
    plt.title(f"t-SNE Embedding of KNN Graph (k={k}, perplexity={perplexity})")
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'tsne_embedding_k{k}_p{perplexity}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export node positions
    positions = {}
    for node, coords in pos.items():
        original_id = G.nodes[node]['original_id']
        positions[original_id] = {
            'x': float(coords[0]),
            'y': float(coords[1]),
            'node_id': node,
            'island': None
        }
        
        # Add island information
        for i, island in enumerate(islands):
            if node in island:
                positions[original_id]['island'] = i
                break
    
    with open(os.path.join(output_dir, f'tsne_positions_k{k}_p{perplexity}.json'), 'w') as f:
        json.dump(positions, f, indent=2)
        
    print(f"t-SNE positions exported to {os.path.join(output_dir, f'tsne_positions_k{k}_p{perplexity}.json')}")
    
    return positions, islands

def analyze_graph(G, islands):
    """Analyze graph properties."""
    results = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_islands": len(islands),
        "island_sizes": [len(island) for island in islands],
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "density": nx.density(G),
    }
    
    # Calculate clustering coefficient (may take time for large graphs)
    if G.number_of_nodes() < 5000:  # Only do this for smaller graphs
        results["clustering_coefficient"] = nx.average_clustering(G)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Visualize KNN Graph Embeddings')
    parser.add_argument('--input', type=str, default='/resnet152_output.json', help='Path to KNN data JSON file')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to use')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--method', type=str, default='both', choices=['force', 'tsne', 'both'], 
                       help='Embedding method to use: force-directed, t-SNE, or both')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    
    args = parser.parse_args()
    
    # Make sure the input path is correct
    if not os.path.exists(args.input):
        if os.path.exists(args.input[1:]):  # Try removing leading slash
            args.input = args.input[1:]
        else:
            print(f"Error: Input file {args.input} not found")
            return
    
    # Load KNN data
    print(f"Loading KNN data from {args.input}...")
    knn_data = load_knn_data(args.input)
    print(f"Loaded {len(knn_data)} data points")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.method in ['force', 'both']:
        # Build and visualize graph using force-directed layout
        print(f"Building graph with k={args.k}...")
        G = build_graph(knn_data, k=args.k)
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        print("Visualizing graph with force-directed layout...")
        pos, islands = visualize_graph(G, args.k, args.output)
        
        # Analyze graph
        print("Analyzing graph properties...")
        analysis = analyze_graph(G, islands)
        
        # Save analysis
        with open(os.path.join(args.output, f'graph_analysis_k{args.k}.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
            
        print(f"Graph analysis saved to {os.path.join(args.output, f'graph_analysis_k{args.k}.json')}")
    
    if args.method in ['tsne', 'both']:
        # Use t-SNE for embedding
        pos, islands = run_tsne_embedding(knn_data, args.k, perplexity=args.perplexity, output_dir=args.output)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()
