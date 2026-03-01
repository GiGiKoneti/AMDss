import logging
from typing import List, Dict, Any
import numpy as np
from .graph_store import LocalGraphStore

logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, graph_store: LocalGraphStore):
        self.graph_store = graph_store
        # Load local tiny LLM for inference simulation
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("GraphRAG loading LLM...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        except Exception as e:
            logger.error(f"Failed to load RAG LLM: {e}")
            self.model = None

    def query(self, user_question: str) -> str:
        logger.info(f"GraphRAG querying: {user_question}")
        
        # 1. Embed the query
        query_vector = self.graph_store.get_embedding(user_question)
        
        # 2. Find seed nodes: top-3 nodes by cosine similarity
        nodes = list(self.graph_store.graph.nodes(data=True))
        if not nodes:
            return "The knowledge graph is currently empty."
            
        similarities = []
        for node_id, data in nodes:
            node_emb = data.get('embedding')
            if node_emb is not None:
                # Cosine similarity
                dot = np.dot(query_vector, node_emb)
                norm = np.linalg.norm(query_vector) * np.linalg.norm(node_emb)
                sim = dot / norm if norm > 0 else 0
                similarities.append((sim, node_id))
        
        similarities.sort(reverse=True)
        seed_nodes = [node_id for sim, node_id in similarities[:3]]
        
        if not seed_nodes:
            return "No embedded nodes found to traverse."

        # 3. Graph traversal: BFS from each seed node, depth=2
        subgraph_nodes = set()
        for seed in seed_nodes:
            import networkx as nx
            ego = nx.ego_graph(self.graph_store.graph, seed, radius=2, undirected=False)
            subgraph_nodes.update(ego.nodes())
            
        subgraph = self.graph_store.graph.subgraph(subgraph_nodes)
        
        # 4. Build context string from traversal
        context_lines = []
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('predicate', data.get('relation', 'related_to'))
            context_lines.append(f'- "{u}" [{rel}] "{v}"')
            
        context_str = "KNOWLEDGE GRAPH CONTEXT:\n" + "\n".join(context_lines)
        if not context_lines:
             context_str += "\nNo strict relationships found."
             
        logger.info(f"Retrieved Graph Context:\n{context_str}")
        
        # 5. Feed this context to the LLM for generation
        if self.model:
            prompt = (
                f"You are a strict technical assistant. Answer the Question using ONLY the information provided in the Context.\n"
                f"If the Context does not contain the answer, you must output 'I do not have enough information to answer that.' DO NOT invent or assume anything.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {user_question}\n"
                f"Answer:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.1, 
                do_sample=False,
                repetition_penalty=1.2
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Format nicely for the UI
            final_output = (
                f"**LLM Generated Answer:**\n{answer}\n\n"
                f"---\n**Graph Traversal Path used:**\n{context_str}"
            )
            return final_output
        else:
            return f"LLM offline. Context retrieved:\n{context_str}"
