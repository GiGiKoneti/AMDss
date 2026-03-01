import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import uuid

from linguist_core.graph_store import LocalGraphStore
from linguist_core.extractor import KnowledgeExtractor
from linguist_core.sync_layer import ZeroMQSyncLayer
from linguist_core.graph_rag import GraphRAG
from linguist_core.models import TripletBroadcast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Linguist-Core Backend API")

NODE_ID = str(uuid.uuid4())[:8]
graph_store = LocalGraphStore()
extractor = KnowledgeExtractor(use_mock=False)

# Read dynamic peer IPs for LAN clustering (e.g. PEER_IPS=192.168.1.5,192.168.1.6)
peer_ips_env = os.environ.get("PEER_IPS", "")
peer_ips_list = [ip.strip() for ip in peer_ips_env.split(",")] if peer_ips_env else []

sync_layer = ZeroMQSyncLayer(node_id=NODE_ID, pub_port=5555, peer_ips=peer_ips_list)
rag_engine = GraphRAG(graph_store)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Linguist-Core node {NODE_ID}")
    def on_broadcast_received(broadcast: TripletBroadcast):
        logger.info(f"Merging knowledge from peer {broadcast.metadata.get('origin_peer_id')}: Node '{broadcast.node_id}'")
        import numpy as np
        # 1. Add the main node with its synced embedding
        emb = np.array(broadcast.embedding, dtype=np.float32)
        if not graph_store.graph.has_node(broadcast.node_id):
            graph_store.graph.add_node(broadcast.node_id, type="Entity", embedding=emb)
        else:
            # Overwrite or average embedding
            graph_store.graph.nodes[broadcast.node_id]["embedding"] = emb
            
        # 2. Add edges
        for edge in broadcast.edges:
            subject, relation, obj = edge
            # Ensure target object exists (without strict embedding for now, or generate fallback)
            if not graph_store.graph.has_node(obj):
                graph_store.graph.add_node(obj, type="Entity", embedding=graph_store.get_embedding(obj))
            graph_store.graph.add_edge(subject, obj, predicate=relation, source_peer=broadcast.metadata.get("origin_peer_id"))
            
    sync_layer.start_listening(on_broadcast_received)

@app.on_event("shutdown")
async def shutdown_event():
    sync_layer.stop()
    graph_store.save()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    raw_content = await file.read()
    filename = file.filename.lower()
    content = ""
    
    try:
        import io
        if filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_content))
            content = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(raw_content))
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            content = raw_content.decode('utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {e}")
        content = raw_content.decode('utf-8', errors='replace')
        
    logger.info(f"Ingesting {len(content)} bytes from {file.filename}")
    
    # Chop content into distinct sentences so the tiny CPU LLM doesn't OOM
    # and has a single, localized concept to extract relations from.
    import re
    chunks = re.split(r'(?<=[.!?])\s+', content)
    
    triplets = []
    for chunk in chunks:
        if len(chunk.strip()) > 20: 
            chunk_triplets = extractor.extract_triplets(chunk, source_ref=file.filename)
            triplets.extend(chunk_triplets)
    
    for t in triplets:
        t.source_node_id = NODE_ID
        
    graph_store.add_triplets(triplets)
    
    def _bg_broadcast():
        import time
        # Group by subject node to sync full nodes
        nodes = {}
        for t in triplets:
            if t.subject not in nodes:
                nodes[t.subject] = []
            nodes[t.subject].append((t.subject, t.predicate, t.object_))
            
        for node_id, edges in nodes.items():
            emb = graph_store.get_embedding(node_id)
            meta = {
                "origin_peer_id": NODE_ID,
                "timestamp": time.time(),
                "source_doc": file.filename
            }
            sync_layer.rcclBroadcast(node_id=node_id, embedding=emb, edges=edges, metadata=meta)

    background_tasks.add_task(_bg_broadcast)
    
    return {"status": "success", "extracted_triplets": len(triplets)}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_graph(req: QueryRequest):
    answer = rag_engine.query(req.query)
    return {"answer": answer}

@app.get("/graph_stats")
async def get_graph_stats():
    return graph_store.get_graph_summary()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
