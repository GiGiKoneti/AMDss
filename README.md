# 🧠 Linguist-Core: Sovereign Distributed Knowledge Graph

Linguist-Core is a high-performance, edge-first Knowledge Graph and RAG (Retrieval-Augmented Generation) engine designed for the **AMD Hackathon 2026**. It allows users to ingest complex documents, extract semantic relationships in real-time, and synchronize that knowledge across a distributed peer-to-peer network—all running locally on AMD hardware.

> [!IMPORTANT]
> This system is designed for **Sovereign AI**. No cloud data, no external APIs. Every computation happens on your silicon.

---

## 🚀 Key Features

*   **Fast Semantic Extraction**: Optimized regex-based relationship extraction that captures technical facts instantly.
*   **Edge-First GraphRAG**: Local Graph Traversal combined with flan-t5 for precise, multi-hop question answering.
*   **Distributed Infinity Fabric Sync**: ZeroMQ-based P2P layer that broadcasts knowledge nodes across your LAN.
*   **Cross-Platform Ready**: Fully compatible with Windows, Linux, and macOS.
*   **AMD Hardware Optimized**: Designed for Ryzen AI NPUs and Radeon GPUs via ROCm.

---

## 🏗 Architecture

1.  **Ingestion Layer**: Parses PDF/DOCX/TXT and splits them into semantic chunks.
2.  **Extraction Engine**: Identifies Entities and meaningful Verbs (e.g., *leverages*, *requires*, *supports*).
3.  **Local Graph Store**: A persistent NetworkX-based multi-directed graph with entity embeddings.
4.  **Sync Layer**: A publisher-subscriber model that replicates graph topology between peers.
5.  **Query Engine**: Performs semantic search + graph BFS to build context for the local RAG LLM.

---

## 💻 Installation

### Prerequisites
- Python 3.9+
- `pip` and `venv`

### 1. Clone & Setup
```bash
git clone https://github.com/GiGiKoneti/AMDss.git
cd AMDss
python3 -m venv venv
```

### 2. Install Dependencies
**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux (Terminal):**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Running the System

To experience the **Distributed Sync**, you should run the system on two separate machines (System A and System B).

### Step 1: Start the API Backend
Identify the IP address of your peer machine.

**System A (at 192.168.0.107):**
```bash
export PEER_IPS="192.168.0.112"
python -m linguist_core.api_server
```

**System B (at 192.168.0.112):**
```bash
export PEER_IPS="192.168.0.107"
python -m linguist_core.api_server
```

### Step 2: Launch the UI 
In a new terminal window (with venv active):
```bash
python -m linguist_core.ui_app
```
Access the UI at: `http://localhost:7860`

---

## 🛠 Advanced Configuration

### Toggling Extraction Mode
In `linguist_core/extractor.py`, you can toggle between **Fast Mode** (Default) and **LLM Mode** (Experimental):
- **Fast Mode**: Immediate results using semantic regex.
- **LLM Mode**: High-accuracy extraction using flan-t5 (Requires GPU/NPU for performance).

---

## 📄 License
Tailored for the AMD Slingshot Hackathon 2026. Built with ❤️ for Sovereign AI.
