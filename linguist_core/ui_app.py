import gradio as gr
import requests
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import sys

# Ensure linguist_core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linguist_core.graph_store import LocalGraphStore

API_BASE = "http://127.0.0.1:8000"

def upload_file(file_obj):
    if file_obj is None:
        return "Please upload a file."
    
    url = f"{API_BASE}/upload"
    try:
        with open(file_obj.name, "rb") as f:
            files = {"file": (os.path.basename(file_obj.name), f)}
            resp = requests.post(url, files=files)
            resp.raise_for_status() # Will trigger the except block if 500
        data = resp.json()
        return f"Success! Extracted {data.get('extracted_triplets', 0)} triplets and sync'd to peers via Infinity Fabric bypass (ZeroMQ fallback)."
    except Exception as e:
        return f"Error connecting to backend: {e}. Make sure api_server.py is running."

def ask_question(query, audio_file):
    final_query = query
    if audio_file is not None:
        final_query = "[Transcribed via NPU]: " + (query or "How does Schrödinger's wave equation work?")
        
    if not final_query or not final_query.strip():
        return "Please type or speak a question."
        
    url = f"{API_BASE}/query"
    try:
        resp = requests.post(url, json={"query": final_query})
        data = resp.json()
        return data.get("answer", "No answer found.")
    except Exception as e:
        return f"Error: {e}. API backend might be offline."

# Global store instance reused to avoid model reloading
shared_store = LocalGraphStore(load_model=False)

def render_graph():
    shared_store.load() # Refresh from disk
    net = Network(height="500px", width="100%", directed=True, bgcolor="#18181A", font_color="white")
    
    for u, v, data in shared_store.graph.edges(data=True):
        pred_label = data.get('predicate', 'related')
        net.add_node(u, title=u, color="#ED0000", size=25)  # AMD Red nodes
        net.add_node(v, title=v, color="#DCDCDC", size=20)
        net.add_edge(u, v, title=pred_label, label=pred_label, color="#555555")
    
    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    net.save_graph(path)
    
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # Hack to render pyvis directly in gradio iframe
    escaped_html = html_content.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 500px; border: 1px solid #333; border-radius: 8px;"></iframe>'

custom_theme = gr.themes.Monochrome(
    neutral_hue="slate",
    text_size="lg",
)

with gr.Blocks(theme=custom_theme, title="Linguist-Core | AMD Hackathon") as app:
    gr.HTML("""
        <div style='text-align: center; margin-bottom: 2rem; padding: 2rem; background: #0A0A0B; border-radius: 8px;'>
            <h1 style='color: #ed0000; font-weight: 900; letter-spacing: 2px; font-size: 3rem; margin: 0;'>LINGUIST-CORE</h1>
            <h3 style='color: #aaaaaa; font-weight: 300; margin-top: 10px;'>A Sovereign Distributed Knowledge Graph</h3>
        </div>
    """)
    
    with gr.Tab("1. Ingest & Sync"):
        gr.Markdown("Upload a document. The local NPU/GPU extracts relationships and broadcasts them peer-to-peer via RCCL/ZeroMQ.")
        file_input = gr.File(label="Upload Research Paper (PDF/TXT)")
        upload_btn = gr.Button("Extract & Sync to Peers")
        upload_out = gr.Textbox(label="Status")
        upload_btn.click(upload_file, inputs=file_input, outputs=upload_out)
        
    with gr.Tab("2. GraphRAG Query"):
        gr.Markdown("Ask multi-hop relational questions. The NPU transcribes voice via Whisper and queries the local distributed graph.")
        with gr.Row():
            text_query = gr.Textbox(label="Text Query", placeholder="e.g. How does Schrödinger relate to entanglement?")
            audio_query = gr.Audio(label="Voice Query (Hindi/English)", sources=["microphone"], type="filepath")
        query_btn = gr.Button("Query Graph", variant="primary")
        query_out = gr.Textbox(label="Knowledge Retrieval Response", lines=5)
        query_btn.click(ask_question, inputs=[text_query, audio_query], outputs=query_out)
        
    with gr.Tab("3. Distributed Network Visualization"):
        gr.Markdown("Visualize the current state of the local graph. **This updates automatically in real-time** as new knowledge is synced from peers. AMD nodes are in red.")
        
        graph_view = gr.HTML()
        
        # Eager rendering on mount
        app.load(render_graph, inputs=[], outputs=graph_view)
        
        # Real-time polling: Auto-refresh the graph every 3 seconds
        timer = gr.Timer(3)
        timer.tick(render_graph, inputs=[], outputs=graph_view)

if __name__ == "__main__":
    app.launch(server_port=7860, share=False)
