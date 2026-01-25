from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SentenceRole:
    """Represents a sentence with its identified role."""
    text: str
    role: str  # topic, definition, example, conclusion, evidence, procedural, imperative
    pos_tags: List[str] = field(default_factory=list)
    is_imperative: bool = False
    semantic_confidence: float = 1.0     
    original_role: Optional[str] = None  
    was_smoothed: bool = False           
    smoothing_reason: str = ""           

@dataclass
class Reference:
    """Represents a reference to a Figure/Table/Equation within a chunk."""
    ref_text: str              
    start_offset: int          
    end_offset: int            
    target_segment_id: str     
    target_type: str           
    ref_kind: str              
    confidence: float = 0.0    

@dataclass
class EnrichedChunk:
    """Standardized output format with references support."""
    chunk_id: str
    heading_path: str
    chunk_type: str
    content: str
    context_prefix: str = ""  
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_segments: List[str] = field(default_factory=list)
    page_range: List[int] = field(default_factory=list)
    depth: int = 0
    word_count: int = 0
    is_cross_page: bool = False
    continuation_type: str = "none"  
    needs_review: bool = False
    merge_evidence: Dict[str, Any] = field(default_factory=dict)
    references: List[Reference] = field(default_factory=list)
    role_distribution: Dict[str, int] = field(default_factory=dict)
    primary_logic_path: str = ""
    semantic_coherence: float = 1.0
    split_reason: str = "none"  
    is_sidebar: bool = False
    sidebar_type: Optional[str] = None
