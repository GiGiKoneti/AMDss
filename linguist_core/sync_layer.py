import zmq
import threading
import json
import time
import logging
import numpy as np
from typing import List, Callable
from .models import TripletBroadcast, KnowledgeTriplet

logger = logging.getLogger(__name__)

class ZeroMQSyncLayer:
    def __init__(self, node_id: str, pub_port: int = 5555, peer_ips: List[str] = None):
        self.node_id = node_id
        self.pub_port = pub_port
        self.peer_ips = peer_ips or []
        
        self.context = zmq.Context()
        
        # Publisher socket (acting as broadcast)
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")
        
        # Subscriber socket
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all
        
        for peer_ip in self.peer_ips:
            try:
                self.sub_socket.connect(f"tcp://{peer_ip}:5555")
                logger.info(f"Connected to peer at {peer_ip}:5555")
            except Exception as e:
                logger.error(f"Failed to connect to peer {peer_ip}: {e}")
            
        self.on_receive_callback = None
        self._running = False
        self._listener_thread = None

    def start_listening(self, callback: Callable[[TripletBroadcast], None]):
        self.on_receive_callback = callback
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        logger.info(f"Node {self.node_id} listening on sub ports {self.sub_ports}.")

    def _listen_loop(self):
        while self._running:
            try:
                # Use poll so we don't block forever
                if self.sub_socket.poll(1000):
                    message = self.sub_socket.recv_string()
                    data = json.loads(message)
                    broadcast = TripletBroadcast(**data)
                    
                    if broadcast.metadata.get("origin_peer_id", "") != self.node_id:
                        logger.info(f"Received sync for node {broadcast.node_id} from {broadcast.metadata.get('origin_peer_id')}")
                        if self.on_receive_callback:
                            self.on_receive_callback(broadcast)
            except Exception as e:
                logger.error(f"Error receiving broadcast: {e}")
                
    def rcclBroadcast(self, node_id: str, embedding: np.ndarray, edges: List[tuple], metadata: dict):
        """Wrapper for RCCL broadcast syncing entire knowledge nodes."""
        payload = TripletBroadcast(
            node_id=node_id,
            embedding=embedding.tolist(),
            edges=[list(e) for e in edges],
            metadata=metadata
        )
        message = payload.model_dump_json()
        self.pub_socket.send_string(message)
        logger.info(f"rcclBroadcast: Synced node {node_id} with {len(edges)} edges.")

    def stop(self):
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=2)
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term()
