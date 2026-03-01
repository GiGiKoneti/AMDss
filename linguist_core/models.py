from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid

class KnowledgeTriplet(BaseModel):
    subject: str = Field(description="The source entity of the relationship")
    predicate: str = Field(description="The relationship type or action")
    object_: str = Field(description="The target entity of the relationship", alias="object")
    source_node_id: Optional[str] = Field(default=None, description="The ID of the peer that broadcasted this")
    source_reference: Optional[str] = Field(default=None, description="The document or source text where this was found")

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TripletBroadcast(BaseModel):
    """Payload sent over the P2P sync bus."""
    node_id: str
    embedding: List[float] # serializable np.ndarray
    edges: List[List[str]] # [(subject, relation, object), ...]
    metadata: Dict[str, Any]
