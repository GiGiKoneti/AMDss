import os
import sys

# Ensure linguist_core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import networkx as nx
from linguist_core.graph_store import LocalGraphStore
from linguist_core.extractor import KnowledgeExtractor

def diagnose_graph(G: nx.DiGraph):
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_weakly_connected_components(G)}")
    
    # CRITICAL CHECK: If average degree < 1.5, your graph is a list of isolated nodes, not a graph
    degrees = [d for _, d in G.degree()]
    print(f"Average degree: {sum(degrees)/len(degrees) if degrees else 0:.2f}")
    
    # Show 5 sample edges with their relationship types
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"  {u} --[{data.get('predicate', data.get('relation', 'NO RELATION TYPE'))}]--> {v}")
    
    # CRITICAL CHECK: Are embeddings attached to nodes?
    sample_node = list(G.nodes(data=True))[0] if G.nodes else None
    print(f"Sample node has embedding: {'embedding' in (sample_node[1] if sample_node else {})}")

if __name__ == "__main__":
    # Remove existing graph to ensure clean test
    if os.path.exists("local_graph.pkl"):
        os.remove("local_graph.pkl")
        
    graph_store = LocalGraphStore()
    extractor = KnowledgeExtractor(use_mock=False)
    
    test_input = "Newton's Second Law states that Force equals mass multiplied by acceleration. This principle underlies rocket propulsion, where thrust force acts on the rocket's mass to produce acceleration."
    
    triplets = extractor.extract_triplets(test_input)
    graph_store.add_triplets(triplets)
    
    print("--- STEP 2 DIAGNOSTIC RESULTS ---")
    diagnose_graph(graph_store.graph)
    print("---------------------------------")
