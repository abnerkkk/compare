import networkx as nx
import random
import matplotlib.pyplot as plt

def create_random_dag(num_nodes, edge_prob):
    """Create a random directed acyclic graph (DAG)"""
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
        for j in range(i):
            if random.random() < edge_prob:
                G.add_edge(j, i)
    return G

def add_edge_attributes(G):
    """Add selection probabilities to edges"""
    for u, v in G.edges():
        G.edges[u, v]['probability'] = random.random()

def add_node_attributes(G, num_attrib_nodes):
    """Add random numerical attributes to nodes"""
    for node in random.sample(list(G.nodes()), num_attrib_nodes):
        G.nodes[node]['attribute'] = random.randint(1, 100)

# Parameters
num_nodes = 10  # total number of nodes
edge_prob = 0.3  # probability of edge creation
num_attrib_nodes = 5  # number of nodes with numerical attributes

# Create DAG
G = create_random_dag(num_nodes, edge_prob)

# Add attributes
add_edge_attributes(G)
add_node_attributes(G, num_attrib_nodes)

# Add subset attribute for multipartite_layout.
# This assigns each node to a vertical layer based on its index.
for node in G.nodes():
    G.nodes[node]["subset"] = node

# Draw the graph
pos = nx.multipartite_layout(G, subset_key="subset")  # positions for all nodes
nx.draw(G, pos, with_labels=True)

# Format edge labels to display probabilities as strings with 2 decimal places
edge_probabilities = nx.get_edge_attributes(G, 'probability')
edge_labels = {edge: f"{prob:.2f}" for edge, prob in edge_probabilities.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()
