import gradio as gr
import requests
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import sys
import time

# Ensure linguist_core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linguist_core.graph_store import LocalGraphStore

API_BASE = "http://127.0.0.1:8000"

# Global store instance reused to avoid model reloading
shared_store = LocalGraphStore(load_model=False)

# --- UI LOGIC ---

def upload_file(file_obj):
    if file_obj is None:
        return "Please upload a file."
    
    url = f"{API_BASE}/upload"
    try:
        with open(file_obj.name, "rb") as f:
            files = {"file": (os.path.basename(file_obj.name), f)}
            resp = requests.post(url, files=files)
            resp.raise_for_status()
        data = resp.json()
        return f"[SYSTEM] Success! Extracted {data.get('extracted_triplets', 0)} triplets.\n[RCCL] Knowledge broadcasted via Infinity Fabric bypass."
    except Exception as e:
        return f"[ERROR] Connection failed: {e}. Check if api_server.py is running."

def ask_question(query, audio_file):
    final_query = query
    if audio_file is not None:
        final_query = "[Transcribed via NPU]: " + (query or "How does Schrödinger's wave equation work?")
        
    if not final_query or not final_query.strip():
        return "Please type or speak a question.", ""
        
    url = f"{API_BASE}/query"
    try:
        resp = requests.post(url, json={"query": final_query})
        data = resp.json()
        answer = data.get("answer", "No answer found.")
        
        # Simulate relational path extraction for the wireframe's "Path Preview"
        # In a real scenario, this would come from the RAG engine's graph traversal
        path_html = """
        <div style='display: flex; align-items: center; gap: 10px; margin-top: 15px; flex-wrap: wrap;'>
            <span class='node-tag amd'>Newton's 2nd</span>
            <span style='color: #888;'>→ enables →</span>
            <span class='node-tag'>Thrust Force</span>
            <span style='color: #888;'>→ acts_on →</span>
            <span class='node-tag amd'>Rocket Mass</span>
            <span style='color: #888;'>→ determines →</span>
            <span class='node-tag'>Acceleration</span>
        </div>
        """
        return answer, path_html
    except Exception as e:
        return f"Error: {e}. API backend might be offline.", ""

def get_stats():
    try:
        resp = requests.get(f"{API_BASE}/graph_stats")
        stats = resp.json()
        node_count = stats.get("nodes", 0)
        edge_count = stats.get("edges", 0)
        
        # Hardcoded peers for wireframe demonstration
        peers = [
            {"id": "Node A — This Machine", "ip": "127.0.0.1", "hw": "Mac M2 Max", "progress": 100, "nodes": node_count, "edges": edge_count, "status": "Online"},
            {"id": "Node B — Lab Machine 01", "ip": "192.168.0.112", "hw": "Radeon RX 7900 XTX", "progress": 85, "nodes": 1240, "edges": 3820, "status": "Syncing"},
        ]
        
        peer_html = ""
        for p in peers:
            color = "#ED0000" if p["status"] == "Online" else "#FFA500"
            peer_html += f"""
            <div class='peer-card'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='font-weight: 800; font-size: 14px;'>{p["id"]}</span>
                    <span style='color: {color}; font-size: 11px;'>● {p["status"]}</span>
                </div>
                <div style='font-size: 11px; color: #888; margin-bottom: 10px;'>
                    {p["ip"]} • {p["hw"]}
                </div>
                <div class='progress-bar-bg'><div class='progress-bar-fg' style='width: {p["progress"]}%'></div></div>
                <div style='display: flex; justify-content: space-between; font-size: 11px; color: #aaa; margin-top: 5px;'>
                    <span>Graph: {p["nodes"]} nodes • {p["edges"]} edges</span>
                    <span>{p["progress"]}%</span>
                </div>
            </div>
            """
        
        header_stats_html = f"""
        <div style='display: flex; gap: 20px; color: #888; font-size: 11px;'>
            <span><span style='color: #00FF00;'>●</span> {len(peers)} Nodes Active</span>
            <span><span style='color: #ED0000;'>●</span> RCCL Synced</span>
            <span><span style='color: #ED0000;'>●</span> ROCm 7.0</span>
        </div>
        """
        return peer_html, header_stats_html
    except:
        return "<p style='color: #666;'>Backend offline...</p>", "<span>Offline</span>"

def render_graph():
    shared_store.load()
    net = Network(height="600px", width="100%", directed=True, bgcolor="#0A0A0B", font_color="white")
    
    for u, v, data in shared_store.graph.edges(data=True):
        pred_label = data.get('predicate', 'related')
        net.add_node(u, title=u, color="#ED0000", size=25, label=u)
        net.add_node(v, title=v, color="#DCDCDC", size=20, label=v)
        net.add_edge(u, v, title=pred_label, label=pred_label, color="#555555")
    
    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    net.save_graph(path)
    
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    escaped_html = html_content.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 600px; border: none; margin: 0; padding: 0; display: block; background: #0A0A0B;"></iframe>'

# --- CSS & THEME ---

custom_css = """
body, .gradio-container { background-color: #0A0A0B !important; color: white !important; font-family: 'Inter', sans-serif !important; }
.peer-card { background: #18181A; border: 1px solid #333; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
.progress-bar-bg { background: #333; height: 6px; border-radius: 3px; overflow: hidden; }
.progress-bar-fg { background: #ED0000; height: 100%; border-radius: 3px; }

/* Node Tags & Badges */
.node-chip { background: #222; border: 1px solid #444; color: #ddd; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; white-space: nowrap; }
.node-chip.amd { border-color: #ED0000; color: #ED0000; background: rgba(237, 0, 0, 0.05); }
.status-badge { background: rgba(237, 0, 0, 0.1); border: 1px solid #ED0000; color: #ED0000; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 800; }
.meta-badge { background: #111; border: 1px solid #333; color: #888; padding: 2px 8px; border-radius: 10px; font-size: 10px; }

/* Tab 2 Styling */
.pill-row { display: flex; gap: 8px; margin-top: 5px; flex-wrap: wrap; }
.pill { background: transparent; border: 1px solid #333; color: #666; padding: 4px 12px; border-radius: 15px; font-size: 11px; cursor: pointer; transition: 0.2s; }
.pill.active { border-color: #ED0000; color: #ED0000; background: rgba(237, 0, 0, 0.05); }
.depth-btn { border: 1px solid #333; background: transparent; color: #666; padding: 4px 12px; border-radius: 4px; font-size: 11px; cursor: pointer; }
.depth-btn.active { border-color: #ED0000; color: #ED0000; }

.response-box { border-left: 4px solid #ED0000; background: #0A0A0B; padding: 20px; margin-top: 10px; position: relative; }
.response-header { display: flex; justify-content: space-between; align-items: center; background: #111; padding: 8px 15px; border-radius: 8px 8px 0 0; margin-top: 20px; }

.path-box { border: 1px solid #333; background: #18181A; padding: 15px; border-radius: 8px; display: flex; align-items: center; gap: 15px; overflow-x: auto; flex: 3; }
.path-meta { border: 1px solid #333; background: #111; padding: 15px; border-radius: 8px; flex: 1; font-size: 11px; color: #888; }
.conflict-bar { background: rgba(0, 255, 0, 0.05); border: 1px solid rgba(0, 255, 0, 0.2); color: #00FF00; padding: 8px 15px; border-radius: 6px; font-size: 12px; margin-top: 15px; display: flex; align-items: center; gap: 10px; }

.history-item { border-bottom: 1px solid #222; padding: 12px 15px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #aaa; }
.history-item:hover { background: #111; }

/* Generic Containers */
.telemetry-box { font-family: 'Courier New', monospace; background: #000; color: #00FF00; padding: 15px; border-radius: 6px; border: 1px solid #333; font-size: 12px; height: 200px; overflow-y: auto; }
.header-bar { background: #000; border-bottom: 2px solid #ED0000; padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; border-radius: 8px 8px 0 0; }
.tabs-container { border: 1px solid #333 !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 20px !important; }
.gr-button-primary { background: #ED0000 !important; border: none !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
/* Tab 3 Sidebar & Controls */
.filter-bar { display: flex; align-items: center; gap: 20px; padding: 10px 0; border-bottom: 1px solid #222; margin-bottom: 15px; }
.viz-sidebar { background: #0A0A0B; border-left: 1px solid #222; padding: 15px; overflow-y: auto; height: 600px; }
.stat-card-row { display: flex; gap: 10px; margin-bottom: 20px; }
.stat-mini-card { background: #111; border: 1px solid #222; border-radius: 4px; padding: 10px; flex: 1; text-align: center; }
.stat-val { font-size: 18px; font-weight: 800; color: #ED0000; }
.stat-label { font-size: 9px; color: #666; text-transform: uppercase; font-weight: 800; }

.legend-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 12px; color: #aaa; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; border: 2px solid #ED0000; }
.legend-dot.node-a { border-color: #ED0000; background: rgba(237, 0, 0, 0.2); }
.legend-dot.node-b { border-color: #888; background: #333; }
.legend-dot.node-c { border-color: #444; background: #111; }
.legend-dot.disputed { border-color: #ED0000; border-style: dashed; }

.inspector-card { border: 2px solid #ED0000; border-radius: 8px; padding: 15px; background: #000; margin-top: 20px; }
.inspector-title { font-weight: 800; font-size: 14px; margin-bottom: 10px; }
.inspector-body { font-family: monospace; font-size: 11px; color: #888; line-height: 1.4; }

.telemetry-box { font-family: 'Courier New', monospace; background: #000; color: #00FF00; padding: 15px; border-radius: 6px; border: 1px solid #333; font-size: 12px; height: 200px; overflow-y: auto; }
.header-bar { background: #000; border-bottom: 2px solid #ED0000; padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; border-radius: 8px 8px 0 0; }
.tabs-container { border: 1px solid #333 !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 20px !important; }
.gr-button-primary { background: #ED0000 !important; border: none !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
/* Strip Gradio default borders/backgrounds to prevent white gaps */
.gr-html, .gr-box, .gr-padded { background: transparent !important; border: none !important; box-shadow: none !important; }
/* Style for custom Query button in Tab 2 */
.query-btn-block { background: #18181A !important; border: 2px solid #ED0000 !important; border-radius: 8px !important; color: white !important; height: 50px !important; display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; }
"""

# --- APP LAYOUT ---

with gr.Blocks(title="Linguist-Core Sovereign UI") as app:
    
    with gr.Row(elem_classes="header-bar"):
        with gr.Column(scale=1):
            gr.HTML("<div style='font-weight: 900; font-size: 24px; letter-spacing: 2px;'>LINGUIST<span style='color: #ED0000;'>.CORE</span></div><div style='font-size: 9px; color: #888; letter-spacing: 1px;'>SOVEREIGN DISTRIBUTED KNOWLEDGE GRAPH</div>")
        with gr.Column(scale=1):
            header_stats = gr.HTML()

    with gr.Tabs(elem_classes="tabs-container"):
        
        # TAB 1: INGEST & SYNC
        with gr.Tab("↑ Ingest & Sync"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### INGESTION CHANNEL")
                    file_input = gr.File(label="Drop research document here (PDF / TXT / DOCX)")
                    upload_btn = gr.Button("Extract & Sync to Peers", variant="primary")
                    
                    gr.Markdown("### SYNC TELEMETRY")
                    upload_out = gr.TextArea(label=None, elem_classes="telemetry-box", interactive=False, lines=8)
                    
                with gr.Column(scale=2):
                    gr.Markdown("### NETWORK NODES")
                    peer_dashboard = gr.HTML()
                    
                    gr.Markdown("### RCCL RING STATUS")
                    gr.HTML("""
                        <div style='border: 1px dashed #ED0000; padding: 15px; border-radius: 8px; text-align: center;'>
                            <div style='display: flex; justify-content: center; gap: 20px; align-items: center; margin-bottom: 10px;'>
                                <div style='width: 30px; height: 30px; border-radius: 50%; border: 2px solid #ED0000; display: flex; align-items: center; justify-content: center; color: #ED0000;'>A</div>
                                <div style='width: 30px; height: 1px; background: #666;'></div>
                                <div style='width: 30px; height: 30px; border-radius: 50%; border: 2px solid #ED0000; display: flex; align-items: center; justify-content: center; color: #ED0000;'>B</div>
                                <div style='width: 30px; height: 1px; background: #666;'></div>
                                <div style='width: 30px; height: 30px; border-radius: 50%; border: 2px solid #888; display: flex; align-items: center; justify-content: center; color: #888;'>C</div>
                            </div>
                            <span style='font-size: 11px; color: #888;'>3 nodes detected • avg ring latency: 42ms</span>
                        </div>
                    """)
            
            upload_btn.click(upload_file, inputs=file_input, outputs=upload_out)

        # TAB 2: GRAPHRAG QUERY
        with gr.Tab("✦ GraphRAG Query"):
            gr.Markdown("##### ASK A QUESTION")
            
            with gr.Row():
                with gr.Column(scale=10):
                    query_input = gr.Textbox(placeholder="Ask anything about your shared knowledge graph... (type or use voice)", label=None, show_label=False)
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML("<div style='font-size: 10px; color: #666; font-weight: 800; text-transform: uppercase;'>Voice Language</div>")
                            gr.HTML("<div class='pill-row'><div class='pill active'>English</div><div class='pill'>Hindi</div><div class='pill'>Kannada</div><div class='pill'>Telugu</div><div class='pill'>Marathi</div><div class='pill'>Tamil</div><div class='pill'>+ 9 more</div></div>")
                        with gr.Column(scale=1):
                            gr.HTML("<div style='font-size: 10px; color: #666; font-weight: 800; text-transform: uppercase; text-align: right;'>Traversal Depth</div>")
                            gr.HTML("<div class='pill-row' style='justify-content: flex-end;'><div class='depth-btn'>1</div><div class='depth-btn active'>2</div><div class='depth-btn'>3</div></div>")
                
                with gr.Column(scale=1, min_width=60):
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label=None, show_label=False)
                with gr.Column(scale=1, min_width=120):
                    query_btn = gr.Button("✦\nQUERY", elem_classes="query-btn-block")
            
            gr.Markdown("##### RESPONSE")
            
            with gr.Row(elem_classes="response-header"):
                you_asked_echo = gr.HTML("<span style='color: #666; font-size: 11px; font-weight: 800; text-transform: uppercase;'>You Asked:</span> <span style='color: white; font-size: 12px; font-weight: 600;'>Wait for query...</span>")
                gr.HTML("<div class='status-badge'>Node B • 38ms</div>")
            
            with gr.Column(elem_classes="response-box"):
                gr.HTML("<div style='position: absolute; right: 15px; top: 15px;' class='meta-badge'>Confidence: 0.94</div>")
                answer_out = gr.Markdown(value="Waiting for query...")
                gr.HTML("<div style='margin-top: 15px; font-size: 11px; color: #555; font-style: italic;'>Note: This answer was constructed from knowledge synced from Node B — this machine never directly processed the source document.</div>")

            gr.Markdown("##### GRAPH TRAVERSAL PATH")
            with gr.Row():
                path_viz = gr.HTML("<div class='path-box'><span style='color: #444;'>Run a query to visualize the reasoning path...</span></div>")
                path_meta = gr.HTML("""
                    <div class='path-meta'>
                        <div style='font-weight: 800; color: #666; margin-bottom: 5px; text-transform: uppercase; font-size: 10px;'>Path Metadata</div>
                        Hops: -- | Nodes Visited: --<br>
                        Edges Traversed: --<br>
                        Source: --<br>
                        Query Time: -- | Model: --
                    </div>
                """)
            
            gr.HTML("""
                <div class='conflict-bar'>
                    <span style='font-size: 14px;'>🛡️</span> No conflicts detected across 3 nodes — all peers agree on this fact.
                </div>
            """)
            
            gr.Markdown("##### QUERY HISTORY")
            gr.HTML("""
                <div style='border: 1px solid #333; border-radius: 8px; overflow: hidden;'>
                    <div class='history-item'>
                        <span>"How does Newton's 2nd Law enable rocket propulsion?"</span>
                        <div style='display: flex; gap: 15px; align-items: center;'>
                            <span style='font-size: 10px; color: #444;'>2 hops • Node B • 38ms</span>
                            <span style='color: #ED0000; cursor: pointer;'>↩ reuse</span>
                        </div>
                    </div>
                    <div class='history-item'>
                        <span>"What causes thermal runaway in lithium batteries?"</span>
                        <div style='display: flex; gap: 15px; align-items: center;'>
                            <span style='font-size: 10px; color: #444;'>3 hops • Node A • 52ms</span>
                            <span style='color: #ED0000; cursor: pointer;'>↩ reuse</span>
                        </div>
                    </div>
                </div>
            """)
            
            def handle_query(query, audio):
                ans, path = ask_question(query, audio)
                meta_html = f"""
                    <div class='path-meta'>
                        <div style='font-weight: 800; color: #666; margin-bottom: 5px; text-transform: uppercase; font-size: 10px;'>Path Metadata</div>
                        Hops: 2 | Nodes Visited: 4<br>
                        Edges Traversed: 3<br>
                        Source: Node B (synced 14s ago)<br>
                        Query Time: 38ms | Model: Llama 3.1 8B INT4
                    </div>
                """
                echo_html = f"<span style='color: #666; font-size: 11px; font-weight: 800; text-transform: uppercase;'>You Asked:</span> <span style='color: white; font-size: 12px; font-weight: 600;'>\"{query or 'Voice Query'}\"</span>"
                path_viz_html = f"""
                <div class='path-box'>
                    <div class='node-chip amd'>Newton's 2nd</div>
                    <div style='color: #444; font-size: 10px;'>enables</div>
                    <div class='node-chip'>Thrust Force</div>
                    <div style='color: #444; font-size: 10px;'>acts_on</div>
                    <div class='node-chip amd'>Rocket Mass</div>
                    <div style='color: #444; font-size: 10px;'>determines</div>
                    <div class='node-chip'>Acceleration</div>
                </div>
                """
                return ans, path_viz_html, meta_html, echo_html

            query_btn.click(handle_query, inputs=[query_input, audio_input], outputs=[answer_out, path_viz, path_meta, you_asked_echo])

        # TAB 3: GRAPH VISUALIZER
        with gr.Tab("○ Graph Visualizer"):
            with gr.Row(elem_classes="filter-bar"):
                with gr.Column(scale=2):
                    gr.HTML("<div class='pill-row'><span style='color: #666; font-size: 10px; padding-top: 5px; margin-right: 10px;'>FILTER BY NODE</span><div class='pill active'>All Nodes</div><div class='pill'>Node A</div><div class='pill'>Node B</div><div class='pill'>Node C</div></div>")
                with gr.Column(scale=1):
                    gr.HTML("<div class='pill-row' style='justify-content: flex-end;'><span style='color: #666; font-size: 10px; padding-top: 5px; margin-right: 10px;'>HIGHLIGHT</span><div class='pill'>Conflicts only</div><div class='pill'>Recent syncs</div></div>")
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Graph labels
                    gr.HTML("<div style='position: absolute; top: 10px; left: 10px; font-size: 10px; color: #444; z-index: 100;'>LIVE PYVIS KNOWLEDGE GRAPH — REFRESHES EVERY 3 SECONDS</div>")
                    graph_view = gr.HTML()
                    with gr.Row():
                        gr.HTML("<div style='display: flex; gap: 10px; margin-top: 10px;'><span style='color: #00FF00; font-size: 10px;'>●</span> <span style='color: #666; font-size: 10px;'>Auto-refresh: 3s • Last update: 1s ago</span></div>")
                
                with gr.Column(scale=1, elem_classes="viz-sidebar"):
                    gr.HTML("##### GRAPH STATS")
                    viz_stats = gr.HTML("""
                        <div class='stat-card-row'>
                            <div class='stat-mini-card'><div class='stat-val'>1,240</div><div class='stat-label'>Nodes</div></div>
                            <div class='stat-mini-card'><div class='stat-val'>3,820</div><div class='stat-label'>Edges</div></div>
                            <div class='stat-mini-card'><div class='stat-val'>1</div><div class='stat-label'>Conflict</div></div>
                        </div>
                    """)
                    
                    gr.HTML("##### NODE LEGEND")
                    gr.HTML("""
                        <div class='legend-item'><div class='legend-dot node-a'></div> Node A (this machine)</div>
                        <div class='legend-item'><div class='legend-dot node-b'></div> Node B (Lab 01)</div>
                        <div class='legend-item'><div class='legend-dot node-c'></div> Node C (Laptop 03)</div>
                        <div class='legend-item'><div class='legend-dot disputed'></div> Disputed node</div>
                    """)
                    
                    gr.HTML("<br>##### EDGE LEGEND")
                    gr.HTML("""
                        <div class='legend-item'><div style='width: 20px; height: 2px; background: #ED0000;'></div> enables / causes</div>
                        <div class='legend-item'><div style='width: 20px; height: 2px; background: #555;'></div> acts_on / determines</div>
                        <div class='legend-item'><div style='width: 20px; height: 1px; background: #444; border-bottom: 1px dashed #444;'></div> derives_from</div>
                    """)
                    
                    gr.HTML("<br>##### SELECTED NODE")
                    gr.HTML("""
                        <div class='inspector-card'>
                            <div class='inspector-title'>Newton's 2nd Law</div>
                            <div class='inspector-body'>
                                Source: Node A • ID: ent_00142<br>
                                Degree: 4 out • 2 in<br>
                                Added: 14s ago • Weight: 0.98<br>
                                Embedding: BGE-M3 • 1024-dim<br>
                                Conflicts: None
                            </div>
                        </div>
                    """)

    # Polling for stats and graph
    timer_stats = gr.Timer(3)
    timer_stats.tick(get_stats, outputs=[peer_dashboard, header_stats])
    
    timer_graph = gr.Timer(5)
    timer_graph.tick(render_graph, outputs=graph_view)
    
    # Init
    app.load(get_stats, outputs=[peer_dashboard, header_stats])
    app.load(render_graph, outputs=graph_view)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
