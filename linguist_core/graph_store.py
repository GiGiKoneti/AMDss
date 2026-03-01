import networkx as nx
import os
import pickle
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from .models import KnowledgeTriplet

logger = logging.getLogger(__name__)

class LocalGraphStore:
    def __init__(self, db_path: str = "local_graph.pkl", load_model: bool = False):
        self.db_path = db_path
        self.graph = nx.MultiDiGraph()
        self._embedding_cache = {}
        self.embedder = None
        
        if load_model:
            self._init_embedder()
            
        self.load()

    def _init_embedder(self):
        # Initialize embedding model (using small CPU model instead of BGE-M3 for dev env)
        if self.embedder is not None:
            return
        try:
            from transformers import pipeline
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self.embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            self.embedder = None

    def get_embedding(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedder is None:
            self._init_embedder()

        if self.embedder:
            # feature-extraction returns list of lists (batch, sequence, hidden_size)
            # We mean-pool over the sequence to get a single vector
            try:
                out = self.embedder(text)
                vec = np.mean(out[0], axis=0)
                emb = np.array(vec, dtype=np.float32)
                self._embedding_cache[text] = emb
                return emb
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                return self._fallback_vector(text)
        else:
            return self._fallback_vector(text)

    def _fallback_vector(self, text: str) -> np.ndarray:
        # Fallback random deterministic vector if ML fails
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.randn(384).astype(np.float32)

    def add_triplets(self, triplets: List[KnowledgeTriplet]):
        """Adds multiple knowledge triplets to the graph, attaching embeddings to nodes."""
        for t in triplets:
            # Ensure nodes exist and have embeddings
            if not self.graph.has_node(t.subject):
                self.graph.add_node(t.subject, type="Entity", embedding=self.get_embedding(t.subject))
            if not self.graph.has_node(t.object_):
                self.graph.add_node(t.object_, type="Entity", embedding=self.get_embedding(t.object_))
            
            # Add directed edge with attributes
            self.graph.add_edge(
                t.subject,
                t.object_,
                predicate=t.predicate,
                source_peer=t.source_node_id,
                reference=t.source_reference
            )
        self.save()

    def get_related_triplets(self, entities: List[str], max_hops: int = 1) -> List[Dict[str, Any]]:
        """
        Extracts a localized subgraph around given entities to feed into GraphRAG context.
        """
        subgraph_nodes = set()
        for entity in entities:
            if self.graph.has_node(entity):
                # Get ego graph (neighborhood)
                ego = nx.ego_graph(self.graph, entity, radius=max_hops, undirected=True)
                subgraph_nodes.update(ego.nodes())
        
        # Extract edges from subgraph nodes
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        results = []
        for u, v, data in subgraph.edges(data=True):
            results.append({
                "subject": u,
                "predicate": data.get("predicate", "related_to"),
                "object": v,
                "source_peer": data.get("source_peer", "local")
            })
        return results

    def get_graph_summary(self) -> Dict[str, int]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges()
        }

    def save(self):
        # We save asynchronously or simply dump to file (it's local proto)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.graph = pickle.load(f)
