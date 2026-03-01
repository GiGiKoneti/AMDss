# Linguist-Core

A Sovereign Distributed Knowledge Graph tailored for the AMD Hackathon.

## Components
- **`models.py`**: Pydantic schemas for Knowledge Triplets and peer-to-peer sync payloads.
- **`graph_store.py`**: NetworkX graph engine for storing entities and relationships.
- **`extractor.py`**: LLM pipeline stub (designed for vLLM local extraction).
- **`sync_layer.py`**: ZeroMQ-based peer-to-peer sync simulating AMD's RCCL Infinity Fabric layer over LAN.
- **`graph_rag.py`**: Graph traversal logic for relational query answering.
- **`voice_asr.py`**: ASR stub for offline transcribed queries using `faster-whisper`.
- **`api_server.py`**: FastAPI backend to coordinate components.
- **`ui_app.py`**: Gradio UI showcasing ingestion, GraphRAG querying, and dynamic visualization.

## How to Run (Cross-Platform Deployment)

This repository is self-contained. You can copy the `linguist_core` directory to any Windows, Mac, or Linux machine.

### 1. Environment Setup (Mac / Linux)
Open your terminal and navigate to the folder where you placed the project:
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Core dependencies
pip install fastapi uvicorn pydantic networkx pyzmq pyvis gradio requests python-multipart

# Install ML Document parsers & LLM dependencies
pip install sentence-transformers transformers torch python-docx pypdf
```

### 2. Multi-Node P2P Sync Routing
To test the ZeroMQ Infinity Fabric emulation across two laptops, you must tell each laptop the IP address of its peer.

**On the Windows Laptop (e.g. at IP `192.168.1.100`):**
Open Powershell:
```powershell
$env:PEER_IPS="192.168.1.101" # The IP of the Mac
python linguist_core\api_server.py

# In a second Powershell:
python linguist_core\ui_app.py
```

**On the Mac Laptop (e.g. at IP `192.168.1.101`):**
Open Terminal:
```bash
export PEER_IPS="192.168.1.100" # The IP of the Windows machine
python -m linguist_core.api_server

# In a second Terminal window:
python -m linguist_core.ui_app
```

The Gradio App will be available on both machines at `http://127.0.0.1:7860`. When you upload a document on the Mac, it will automatically embed the nodes and broadcast the NetworkX topology direct to the Windows machine via the TCP Socket on port 5555!
